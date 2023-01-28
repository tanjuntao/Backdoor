import torch.random
from torch import nn


class Embedding(nn.Module):
    def __init__(
        self,
        input_length: int,
        cat_idxes: list,
        cat_size_dict: dict,
        cat_dim_dict: dict = None,
        cat_dropout_rate: float = 0.1,
        random_state: int = None,
    ):
        """

        Parameters
        ----------
        input_length
        cat_idxes
        cat_size_dict
        cat_dim_dict
        cat_dropout_rate
        random_state
        """

        super(Embedding, self).__init__()
        self.cat_idxes = cat_idxes
        self.cont_idxes = list(set(list(range(input_length))) - set(cat_idxes))
        self.n_cat = len(cat_idxes)
        self.n_cont = input_length - self.n_cat

        if random_state is not None:
            torch.random.manual_seed(random_state)
        if cat_dim_dict is None:
            cat_dim_dict = {
                idx: Embedding.embedding_dim_rule(size)
                for idx, size in cat_size_dict.items()
            }

        embeddings = {}
        for idx in cat_idxes:
            embed_size = cat_size_dict[idx]
            embed_dim = cat_dim_dict[idx]
            curr_embedding = nn.Embedding(embed_size, embed_dim)
            embeddings[idx] = curr_embedding
        self.embeddings = nn.ModuleDict(embeddings)
        self.cat_dropout = nn.Dropout(cat_dropout_rate)
        self.cont_bn = nn.BatchNorm1d(self.n_cont)

    def forward(self, x):
        """
        The model architecture is:
               ---- x_cat  ---- embeddings ---- dropout    ----
              |                                                |
        x ----                                               concat ---- final_outputs
              |                                                |
               ---- x_cont -------------------- batch norm ----

        Parameters
        ----------
        x

        Returns
        -------

        """
        x_cont = x[:, self.cont_idxes]

        cat_outputs = []
        for idx in self.cat_idxes:
            cat_in = x[:, idx]
            cat_out = self.embeddings[idx](cat_in)
            cat_outputs.append(cat_out)
        cat_outputs = torch.cat(cat_outputs, dim=1)
        cat_outputs = self.cat_dropout(cat_outputs)
        cont_outputs = self.cont_bn(x_cont)
        final_outputs = torch.cat((cat_outputs, cont_outputs), dim=1)

        return final_outputs

    @staticmethod
    def embedding_dim_rule(size):
        # This rule is adapted from FastAi.
        # ref: https://forums.fast.ai/t/size-of-embedding-for-categorical-variables/42608/2  # noqa: E501
        return min(50, (size + 1) // 2)
