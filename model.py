from torch import nn
import torch
from torch_multi_head_attention import MultiHeadAttention


class _MultiLayerPercep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_MultiLayerPercep, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.ReLU(),            
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)


class _Aggregation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_Aggregation, self).__init__()
        self.aggre = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.aggre(x)


class _UserModel(nn.Module):
    ''' 
    User modeling to learn user latent factors.
    User modeling leverages two types aggregation: item aggregation and social aggregation
    '''

    def __init__(self, emb_dim, user_emb, item_emb, rate_emb):
        super(_UserModel, self).__init__()
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb
        self.emb_dim = emb_dim
        self.num_heads = 16

        self.g_v = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)

        self.user_items_att = _MultiLayerPercep(2 * self.emb_dim, self.num_heads)
        self.user_items_head = _MultiLayerPercep(self.num_heads * 12, 12) #TODO change hardcode
        
        self.aggre_items = _Aggregation(self.emb_dim, self.emb_dim)

        self.user_users_att = _MultiLayerPercep(2 * self.emb_dim, self.num_heads)
        self.user_users_head = _MultiLayerPercep(self.num_heads * 12, 12)

        self.aggre_neigbors = _Aggregation(self.emb_dim, self.emb_dim)
        
        self.combine_mlp = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # used for preventing zero div error when calculating softmax score
        self.eps = 1e-10
        

    def forward(self, uids, u_item_pad, u_user_pad, u_user_item_pad):
        # item aggregation
        q_a = self.item_emb(u_item_pad[:,:,0])   # B x maxi_len x emb_dim
        mask_u = torch.where(u_item_pad[:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))   # B x maxi_len
        u_item_er = self.rate_emb(u_item_pad[:,:,1])  # B x maxi_len x emb_dim
        
        x_ia = self.g_v(torch.cat([q_a, u_item_er], dim = 2).view(-1, 2 * self.emb_dim)).view(q_a.size())  # B x maxi_len x emb_dim
        # mask_u = mask_u.unsqueeze(-1).expand(-1, -1, 12)
        # print("mask size", mask_u.shape)
        ## calculate attention scores in item aggregation 
        # embeds =  self.user_emb(uids).unsqueeze(1).expand_as(x_ia).unsqueeze(-1).expand(-1, -1, -1, 12)
        # print("embed", embeds.shape)
        # p_i = mask_u.unsqueeze(-2).expand(-1, -1, 64, -1) * embeds  # B x maxi_len x emb_dim
        # print("p_i size", p_i.shape)
        p_i = mask_u.unsqueeze(2).expand_as(x_ia) * self.user_emb(uids).unsqueeze(1).expand_as(x_ia)  # B x maxi_len x emb_dim
        # x_ia = x_ia.unsqueeze(-1).expand(-1, -1, -1, 12)
        alpha = self.user_items_att(torch.cat([x_ia, p_i], dim = 2).view(-1, 2 * self.emb_dim))
        mask_un = mask_u.unsqueeze(-1).expand(-1, -1, self.num_heads)
        alpha = alpha.view(mask_un.size()) # B x maxi_len
        alpha_r = alpha.view(-1, alpha.shape[-2] * self.num_heads)
        alpha_r = self.user_items_head(alpha_r)

        alpha = torch.exp(alpha_r) * mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)        
        
        h_iI = self.aggre_items(torch.sum(alpha.unsqueeze(2).expand_as(x_ia) * x_ia, 1))     # B x emb_dim

        # social aggregation
        q_a_s = self.item_emb(u_user_item_pad[:,:,:,0])   # B x maxu_len x maxi_len x emb_dim
        mask_s = torch.where(u_user_item_pad[:,:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))  # B x maxu_len x maxi_len
        u_user_item_er = self.rate_emb(u_user_item_pad[:,:,:,1]) # B x maxu_len x maxi_len x emb_dim
        
        x_ia_s = self.g_v(torch.cat([q_a_s, u_user_item_er], dim = 3).view(-1, 2 * self.emb_dim)).view(q_a_s.size())  # B x maxu_len x maxi_len x emb_dim   

        p_i_s = mask_s.unsqueeze(3).expand_as(x_ia_s) * self.user_emb(u_user_pad).unsqueeze(2).expand_as(x_ia_s)  # B x maxu_len x maxi_len x emb_dim

        alpha_s = self.user_items_att(torch.cat([x_ia_s, p_i_s], dim = 3).view(-1, 2 * self.emb_dim))    # B x maxu_len x maxi_len
        # print("social", alpha_s.shape, mask_s.shape)
        mask_sn = mask_s.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)
        alpha_s = alpha_s.view(mask_sn.size())
        alpha_r = alpha_s.view(alpha_s.shape[0], alpha_s.shape[1], alpha_s.shape[-2] * self.num_heads)
        alpha_r = self.user_items_head(alpha_r)
        alpha_s = torch.exp(alpha_r) * mask_s
        alpha_s = alpha_s / (torch.sum(alpha_s, 2).unsqueeze(2).expand_as(alpha_s) + self.eps)
        # alpha_s = alpha_s.mean(-1)

        h_oI_temp = torch.sum(alpha_s.unsqueeze(3).expand_as(x_ia_s) * x_ia_s, 2)    # B x maxu_len x emb_dim
        h_oI = self.aggre_items(h_oI_temp.view(-1, self.emb_dim)).view(h_oI_temp.size())  # B x maxu_len x emb_dim

        ## calculate attention scores in social aggregation
        beta = self.user_users_att(torch.cat([h_oI, self.user_emb(u_user_pad)], dim = 2).view(-1, 2 * self.emb_dim))
        mask_su = torch.where(u_user_pad > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        # print(beta.shape, mask_su.shape)
        mask_sun = mask_su.unsqueeze(-1).expand(-1, -1, self.num_heads)
        beta = beta.view(mask_sun.size()) # B x maxi_len
        # print(beta.shape)
        beta_r = beta.view(-1, beta.shape[-2] * self.num_heads)
        beta_r = self.user_users_head(beta_r)
        
        beta = torch.exp(beta_r) * mask_su
        beta = beta / (torch.sum(beta, 1).unsqueeze(1).expand_as(beta) + self.eps)
        h_iS = self.aggre_neigbors(torch.sum(beta.unsqueeze(2).expand_as(h_oI) * h_oI, 1))     # B x emb_dim

        ## learning user latent factor
        h_i = self.combine_mlp(torch.cat([h_iI, h_iS], dim = 1))

        return h_i 
        


class _ItemModel_GraphRecPlus(nn.Module):

    # Item modeling to learn item latent factors.
    # Item modeling leverages two types aggregation: user aggregation and item2item aggregation

    def __init__(self, emb_dim, user_emb, item_emb, rate_emb):
        super(_ItemModel_GraphRecPlus, self).__init__()
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb
        self.emb_dim = emb_dim
        self.num_heads = 16


        self.g_u = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)

        # self.item_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)

        self.item_users_att = _MultiLayerPercep(2 * self.emb_dim, self.num_heads)
        self.item_users_head = _MultiLayerPercep(12 * self.num_heads, 12) # TODO

        self.aggre_items = _Aggregation(self.emb_dim, self.emb_dim)

        self.item_items_att = _MultiLayerPercep(2 * self.emb_dim, self.num_heads)
        self.item_items_head = _MultiLayerPercep(12 * self.num_heads, 12) # TODO


        self.aggre_item2item = _Aggregation(self.emb_dim, self.emb_dim)

        self.combine_mlp = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # used for preventing zero div error when calculating softmax score
        self.eps = 1e-10

    def forward(self, iids, i_user_pad, i_item_pad, i_item_user_pad):
        # user aggregation
        p_t = self.user_emb(i_user_pad[:, :, 0])
        mask_i = torch.where(i_user_pad[:, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))
        i_user_er = self.rate_emb(i_user_pad[:, :, 1])

        f_jt = self.g_u(torch.cat([p_t, i_user_er], dim=2).view(-1, 2 * self.emb_dim)).view(p_t.size())

        # calculate attention scores in user aggregation
        q_j = mask_i.unsqueeze(2).expand_as(f_jt) * self.item_emb(iids).unsqueeze(1).expand_as(f_jt)

        miu = self.item_users_att(torch.cat([f_jt, q_j], dim=2).view(-1, 2 * self.emb_dim))
        mask_iu = mask_i.unsqueeze(-1).expand(-1, -1, self.num_heads)
        miu = miu.view(mask_iu.size())
        miu_r = miu.view(miu.shape[0], miu.shape[-2] * self.num_heads)
        miu_r = self.item_users_head(miu_r)

        miu = torch.exp(miu_r) * mask_i
        miu = miu / (torch.sum(miu, 1).unsqueeze(1).expand_as(miu) + self.eps)

        h_jU = self.aggre_items(torch.sum(miu.unsqueeze(2).expand_as(f_jt) * f_jt, 1))

        # social aggregation
        q_a_s = self.item_emb(i_item_user_pad[:, :, :, 0])  # B x maxu_len x maxi_len x emb_dim
        mask_s = torch.where(i_item_user_pad[:, :, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))  # B x maxu_len x maxi_len
        i_item_user_er = self.rate_emb(i_item_user_pad[:, :, :, 1])  # B x maxu_len x maxi_len x emb_dim

        x_ia_s = self.g_u(torch.cat([q_a_s, i_item_user_er], dim=3).view(-1, 2 * self.emb_dim)).view(
            q_a_s.size())  # B x maxu_len x maxi_len x emb_dim
        p_i_s = mask_s.unsqueeze(3).expand_as(x_ia_s) * self.item_emb(i_item_pad).unsqueeze(2).expand_as(
            x_ia_s)  # B x maxu_len x maxi_len x emb_dim

        miu_s = self.item_users_att(torch.cat([x_ia_s, p_i_s], dim=3).view(-1, 2 * self.emb_dim))  # B x maxu_len x maxi_len
        
        mask_sn = mask_s.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)
        miu_s = miu_s.view(mask_sn.size())

        miu_r = miu_s.view(miu_s.shape[0], miu_s.shape[1], miu_s.shape[-2] * self.num_heads)
        miu_r = self.item_users_head(miu_r)

        miu_s = torch.exp(miu_r) * mask_s
        miu_s = miu_s / (torch.sum(miu_s, 2).unsqueeze(2).expand_as(miu_s) + self.eps)

        h_oU_temp = torch.sum(miu_s.unsqueeze(3).expand_as(x_ia_s) * x_ia_s, 2)  # B x maxu_len x emb_dim
        h_oU = self.aggre_items(h_oU_temp.view(-1, self.emb_dim)).view(h_oU_temp.size())  # B x maxu_len x emb_dim

        ####

        kappa = self.item_items_att(torch.cat([h_oU, self.item_emb(i_item_pad)], dim=2).view(-1, 2 * self.emb_dim))
        mask_su = torch.where(i_item_pad > 0, torch.tensor([1.], device=self.device),
                              torch.tensor([0.], device=self.device))

        mask_sun = mask_su.unsqueeze(-1).expand(-1, -1, self.num_heads)
        kappa_s = kappa.view(mask_sun.size())
        # kappa_r = kappa_s.view(kappa_s.shape[0], kappa_s.shape[1], kappa_s.shape[-2] * self.num_heads)
        kappa_r = kappa_s.view(-1, kappa_s.shape[-2] * self.num_heads)

        kappa_r = self.item_items_head(kappa_r)

        kappa = torch.exp(kappa_r) * mask_su
        kappa = kappa / (torch.sum(kappa, 1).unsqueeze(1).expand_as(kappa) + self.eps)
        h_jV = self.aggre_item2item(torch.sum(kappa.unsqueeze(2).expand_as(h_oU) * h_oU, 1))  # B x emb_dim

        ## learning user latent factor
        h_j = self.combine_mlp(torch.cat([h_jU, h_jV], dim=1))

        return h_j


class _ItemModel(nn.Module):
    '''Item modeling to learn item latent factors.
    '''
    def __init__(self, emb_dim, user_emb, item_emb, rate_emb):
        super(_ItemModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb
        self.num_heads = 8


        self.g_u = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        
        self.item_users_att = _MultiLayerPercep(2 * self.emb_dim, self.num_heads)
        self.item_users_head = _MultiLayerPercep(12 * self.num_heads, 12) # TODO 
        self.aggre_users = _Aggregation(self.emb_dim, self.emb_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # used for preventing zero div error when calculating softmax score
        self.eps = 1e-10

    def forward(self, iids, i_user_pad):
        # user aggregation
        p_t = self.user_emb(i_user_pad[:,:,0])
        mask_i = torch.where(i_user_pad[:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        i_user_er = self.rate_emb(i_user_pad[:,:,1])
        
        f_jt = self.g_u(torch.cat([p_t, i_user_er], dim = 2).view(-1, 2 * self.emb_dim)).view(p_t.size())
        
        # calculate attention scores in user aggregation
        q_j = mask_i.unsqueeze(2).expand_as(f_jt) * self.item_emb(iids).unsqueeze(1).expand_as(f_jt)
        # print(mask_i.shape)
        miu = self.item_users_att(torch.cat([f_jt, q_j], dim = 2).view(-1, 2 * self.emb_dim))
        mask_iu = mask_i.unsqueeze(-1).expand(-1, -1, self.num_heads)
        miu = miu.view(mask_iu.size())
        miu_r = miu.view(miu.shape[0], miu.shape[-2] * self.num_heads)
        miu_r = self.item_users_head(miu_r)
        # miu = miu.view(mask_i.size())
        # print(miu.shape)
        
        miu = torch.exp(miu_r) * mask_i
        miu = miu / (torch.sum(miu, 1).unsqueeze(1).expand_as(miu) + self.eps)
        # miu = miu.mean(dim=-1)

        z_j = self.aggre_users(torch.sum(miu.unsqueeze(2).expand_as(f_jt) * f_jt, 1))

        return z_j


class GraphRec(nn.Module):
    '''GraphRec model proposed in the paper Graph neural network for social recommendation 

    Args:
        number_users: the number of users in the dataset.
        number_items: the number of items in the dataset.
        num_rate_levels: the number of rate levels in the dataset.
        emb_dim: the dimension of user and item embedding (default = 64).

    '''
    def __init__(self, num_users, num_items, num_rate_levels, emb_dim = 64, dataset=None):
        super(GraphRec, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_rate_levels = num_rate_levels
        self.emb_dim = emb_dim
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim, padding_idx = 0)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim, padding_idx = 0)
        self.rate_emb = nn.Embedding(self.num_rate_levels, self.emb_dim, padding_idx = 0)

        self.user_model = _UserModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb)

        if dataset == "FilmTrust":
          self.item_model = _ItemModel_GraphRecPlus(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb)
        else:
          self.item_model = _ItemModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb)
        
        self.rate_pred = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 1),
        )


    def forward(self, uids, iids, u_item_pad, u_user_pad, u_user_item_pad, i_user_pad, dataset=None, i_item_pad=None, i_item_user_pad=None):
        '''
        Args:
            uids: the user id sequences.
            iids: the item id sequences.
            u_item_pad: the padded user-item graph.
            u_user_pad: the padded user-user graph.
            u_user_item_pad: the padded user-user-item graph.
            i_user_pad: the padded item-user graph.

        Shapes:
            uids: (B).
            iids: (B).
            u_item_pad: (B, ItemSeqMaxLen, 2).
            u_user_pad: (B, UserSeqMaxLen).
            u_user_item_pad: (B, UserSeqMaxLen, ItemSeqMaxLen, 2).
            i_user_pad: (B, UserSeqMaxLen, 2).

        Returns:
            the predicted rate scores of the user to the item.
        '''

        h_i = self.user_model(uids, u_item_pad, u_user_pad, u_user_item_pad)
        if dataset == "FilmTrust":
          z_j = self.item_model(iids, i_user_pad, i_item_pad, i_item_user_pad)

        else:  
          z_j = self.item_model(iids, i_user_pad)

        # make prediction
        r_ij = self.rate_pred(torch.cat([h_i, z_j], dim = 1))

        return r_ij
