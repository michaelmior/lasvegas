from hyperopt import STATUS_FAIL, STATUS_OK

from lasvegas import eval, constants, train
from lasvegas.state import create_model


def objective(params):
    """
    Train and evaluate the model using the given parameters,
    scoring each model with wins + draws / 2
    """
    # Check that the parameters are valid
    if params['TIE_REWARD'] > params['WIN_REWARD'] or \
       params['MINIBATCH_SIZE'] > params['LEARNING_MEMORY_SIZE']:
        return {'loss': None, 'status': STATUS_FAIL}

    # Convert necessary parameters to integers
    params['LEARNING_MEMORY_SIZE'] = int(params['LEARNING_MEMORY_SIZE'])
    params['MINIBATCH_SIZE'] = int(params['MINIBATCH_SIZE'])

    # Apply all the parameters
    for (key, val) in params.items():
        setattr(constants, key, val)

    model = create_model(kernel_initializer=params['kernel_initializer'],
                         activation=params['activation'],
                         optimizer=params['optimizer'])
    train.train_model(model,
                      params['training_games'] // params['MINIBATCH_SIZE'],
                      'changing')
    stats = eval.eval_model('model', params['eval_games'], 'changing', model)

    return {'loss': -(stats['wins'] + stats['draws'] / 2), 'status': STATUS_OK}
