"""Three separate stages:

    0. Function to generate ground truths:
        Input: model, x_test, test_pred
        Output: 1-D binary array, 1 if classified correctly, zero if classified incorrectly

    1. Function to create sub-models
    Input: model (trained), list of layer numbers, no. of output nodes (num_classes)
    Output: list of sub_models

    2. Function to train sub models (Only final layer connections to fully connected). Freeze other weights.
        Input: list of sub_models, x_train
        Output: list of trained sub_models

        Number of epochs? Unclear, paper does not specify.

    3. Function to calculate prediction profile: One softmax vector per sub-model
        Input: List of trained sub_models
        Outputs: k Array of softmax arrays, one for each sub_model, shape: #classes * #sub_models

    4. Function to calculate SV score
        Input: a) k array of softmax arrays, one for each sub_model, shape: #classes * #sub_models
               b) train_pred: Array of training predictions of original model M

        Output: Array of SV scores of length k, hopefully ordered by shallow to deep sub models.

    5. Function to calculate PV score for all x_test
        Input: a) x_test, array_sv_score, weights_array
        Output: Array of PV scores in (0,1).

    6. Function to calculate weights array [3 options: Logarithmic, Linear, Exponential]:
        Input: a) Type of weight growth
               b) Number k of specific layer

        Output: array of weights, shape: (#sub_models,)

    7. Function calculate AUC score:
        Inputs: a) generate_ground_truth(model, test_pred, x_test)
                b) PV score array
        Output: roc_auc_score(labels, pv_score_array)
    """
