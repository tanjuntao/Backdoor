class _DecisionNode:
    def __init__(
        self,
        party_id=None,
        record_id=None,
        left_branch=None,
        right_branch=None,
        value=None,
        hist_list=None,
        sample_tag_selected=None,
        sample_tag_unselected=None,
        split_party_id=None,
        split_feature_id=None,
        split_bin_id=None,
        split_gain=None,
        depth=None,
    ):
        # node saving variables
        self.party_id = party_id
        self.record_id = record_id
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.value = value

        # temporary training variables
        self.sample_tag_selected = sample_tag_selected
        self.sample_tag_unselected = sample_tag_unselected
        self.hist_list = hist_list
        self.split_party_id = split_party_id
        self.split_feature_id = split_feature_id
        self.split_bin_id = split_bin_id
        self.split_gain = split_gain
        self.depth = depth

    def __lt__(self, other):
        return self.split_gain > other.split_gain
