class Air_Attrs:
    def __init__(self, args):
        self.args = args
        self.adj_mx = args.adj_mx
        self.num_nodes = args.adj_mx.shape[0]
        self.num_edges = args.edge_index.shape[1]

        self.seq_len = int(args.model.seq_len)
        self.horizon = int(args.model.pred_len)
        self.input_dim = int(args.model.input_dim)
        self.X_dim = int(args.model.X_dim)

class EncoderAttrs:
    def __init__(self, args):
        self.args = args
        self.num_rnn_layers = int(args.model.num_rnn_layers)
        self.rnn_dim = int(args.model.rnn_dim)
        self.latent_dim = int(args.model.dim.latent)

        self.seq_len = int(args.model.seq_len)
        self.pred_len = int(args.model.pred_len)
        self.input_dim = int(args.model.input_dim)
        self.output_dim = int(args.model.output_dim)