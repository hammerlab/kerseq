from keras.models import Graph
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense

from .reverse_time import ReverseTime

def _expand_arg_to_dict(
        arg_name, arg_value, expected_type, sequence_input_names):
    if isinstance(arg_value, expected_type):
        return {name: arg_value for name in sequence_input_names}
    elif isinstance(arg_value, dict):
        missing_names = [
            name
            for name in sequence_input_names
            if name not in arg_value.keys()
        ]
        if len(missing_names) > 0:
            raise ValueError("Missing input names %s' for arg %s" % (
                missing_names, arg_name))
        return arg_value
    else:
        raise ValueError(
            "Expected %s to be dict from names to %s, not %s : %s" % (
                arg_name, expected_type, arg_value, type(arg_value)))

def build_rnn_graph(
        sequence_input_names,
        n_symbols,
        rnn_class=LSTM,
        embedding=True,
        embedding_dim=None,
        conv_before_rnn=False,
        conv_filter_size=3,
        conv_maxpool_size=2,
        n_conv_layers=2,
        rnn_output_dim=64,
        rnn_bidirectional=True,
        dense_output_dims=[400],
        dense_activation="relu",
        output_dim=1,
        output_activation="sigmoid",
        optimizer="rmsprop",
        loss="mse"):
    """
    Create an LSTM which expect sequences of indices (starting from 1),
    embeds them in a vector space, and feeds the sequence of vectors to
    an RNN.
    """
    if isinstance(dense_output_dims, int):
        dense_output_dims = [dense_output_dims]
    # change rnn_output_dim to a dictionary mapping input names to
    # RNN output sizes
    rnn_output_dim = _expand_arg_to_dict(
        "rnn_output_dim", rnn_output_dim, int, sequence_input_names)
    conv_before_rnn = _expand_arg_to_dict(
        "conv_before_rnn", conv_before_rnn, bool, sequence_input_names)
    conv_filter_size = _expand_arg_to_dict(
        "conv_filter_size", conv_filter_size, int, sequence_input_names)
    conv_maxpool_size = _expand_arg_to_dict(
        "conv_maxpool_size", conv_maxpool_size, int, sequence_input_names)

    model = Graph()
    rnn_names = []
    rnn_dims = []
    for input_name in sequence_input_names:
        model.add_input(name=input_name, ndim=2, dtype=int)
        if embedding:
            if embedding_dim is None:
                embedding_dim = n_symbols

            embedding_name = input_name + "_embedding"
            model.add_node(
                Embedding(
                    # adding 1 to number of symbols since we're
                    # expecting indices to be base-1
                    input_dim=n_symbols + 1,
                    output_dim=embedding_dim,
                    mask_zero=True),
                name=embedding_name,
                input=input_name)
            input_to_conv = embedding_name
            input_to_conv_dim = embedding_dim
        else:
            input_to_conv = input_name
            input_to_conv_dim = n_symbols

        if conv_before_rnn[input_name]:
            raise ValueError("Masked convolution not yet implemented")
            # model.add_node()
            # conv_name = input_name + "_conv"
            # maxpool_name = conv_name + "_maxpool"
            # input_to_rnn = maxpool_name
        else:
            input_to_rnn = input_to_conv
            input_to_rnn_dim = input_to_conv_dim

        # add RNN for summarizing a sequence input into a vector
        # representation
        rnn_name = input_name + "_rnn"
        rnn_dim = rnn_output_dim[input_name]
        rnn_dims.append(rnn_dim)
        model.add_node(
            rnn_class(
                input_dim=input_to_rnn_dim,
                output_dim=rnn_dim),
            name=rnn_name,
            input=input_to_rnn)
        rnn_names.append(rnn_name)

        if rnn_bidirectional:
            reverse_input_to_rnn = input_to_rnn + "_reverse"
            model.add_node(
                ReverseTime(),
                name=reverse_input_to_rnn,
                input=input_to_rnn)
            # add RNN for reverse sequence
            rnn_reverse_name = rnn_name + "_reverse"
            model.add_node(
                rnn_class(
                    input_dim=input_to_rnn_dim,
                    output_dim=rnn_dim),
                name=rnn_reverse_name,
                input=reverse_input_to_rnn)
            rnn_double_reverse_name = rnn_reverse_name + "_reverse"
            model.add_node(
                ReverseTime(),
                name=rnn_double_reverse_name,
                input=rnn_reverse_name)
            rnn_names.append(rnn_double_reverse_name)
            rnn_dims.append(rnn_dim)
    if len(dense_output_dims) == 0:
        raise ValueError("Dense hidden layer required")

    # concatenate last output of all RNNs and
    # transform them into a lower dimensional space
    for i, dense_output_dim in enumerate(dense_output_dims):
        if i == 0:
            # if we have a list with more than one element, then we merge
            first_dense_layer = Dense(
                sum(rnn_dims),
                dense_output_dim,
                activation=dense_activation)
            if len(rnn_names) > 1:
                model.add_node(
                    first_dense_layer,
                    name="dense1",
                    merge_mode="concat",
                    inputs=rnn_names)
            elif len(rnn_names) == 1:
                # we have one element, so no merge required
                model.add_node(
                    first_dense_layer,
                    name="dense1",
                    input=rnn_names[0])
            else:
                raise ValueError("RNN layer required")
        else:
            model.add_node(
                Dense(
                    dense_output_dims[i - 1],
                    dense_output_dim,
                    activation=dense_activation),
                name="dense%d" % (i + 1),
                input="dense%d" % i)
        # if dropout_probability > 0:
        #    model.add(Dropout(dropout_probability), input_name)

    model.add_node(
        Dense(dense_output_dim, output_dim, activation=output_activation),
        name="final_dense",
        input="dense%d" % len(dense_output_dims))
    model.add_output(name='output', input="final_dense")
    model.compile(optimizer, {"output": loss})
    return model
