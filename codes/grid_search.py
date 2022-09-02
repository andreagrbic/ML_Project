import pandas as pd


def grid_search(model_type, df_user_artists_train: pd.DataFrame, df_user_artists_val: pd.DataFrame,
                df_user_friends: pd.DataFrame, df_user_tagged_artists: pd.DataFrame, params: dict):
    """
        Funkcija u kojoj se vrši pretraga optimalnih hiperparametara modela. Parametri po kojima se model obučava na
        trening skupu za različite vrijednosti hiperparametara su learning_rate, regularization, n_factors.
        Potom se model ocjenjuje na validacionom skupu. Na osnovu ovako dobijenih rezultata se biraju optimalne
        vrijednosti hiperparametara.

        Parameters:
            model_type: Klasa (primjerak klase KPMF ili CKPMF) modela za koji se vrši grid search.
            df_user_artists_train:  Trening skup.
            df_user_artists_val: Skup za validaciju.
            df_user_friends: Skup podataka koji sadrži informacije o korisnicima i njihovim prijateljima.
            df_user_tagged_artists: Skup podataka koji sadrži informacije o izvođačima i dodijeljenim tagovima
             (žanrovima muzike).
            params: Dodatni parametri modela (izbor kernela).

        Returns:
            Optimalni hiperparametri modela koji se koriste za njegovo obučavanje.
    """
    result = {'min_val_loss': 999, 'lr': None, 'reg': None, 'factor': None}

    # Inicijalizacija parametara modela
    lr_list = [0.01, 0.005, 0.001]
    n_epochs_list = [10, 20, 40]
    reg_list = [0.1, 0.05, 0.01]
    factor_list = [4, 8, 32]

    for lr, n_epochs in zip(lr_list, n_epochs_list):
        for reg in reg_list:
            for factor in factor_list:
                # Inicijalizacija modela
                params['learning_rate'] = lr
                params['regularization'] = reg
                params['n_factors'] = factor
                params['n_epochs'] = n_epochs
                model = model_type(**params)

                print("trying learning rate : {}, regularization : {}, n_factors: {}".format(lr, reg, factor))
                model.fit(df_user_artists_train, df_user_artists_val, df_user_friends, df_user_tagged_artists)

                # Provjera da li je model bolji od do sada najboljeg
                if model.val_loss < result['min_val_loss']:
                    result['min_val_loss'] = model.val_loss
                    result['lr'] = lr
                    result['reg'] = reg
                    result['factor'] = factor

    print('best lr: {}, best reg: {}, best n_factors: {}, min val rmse: {}'.format(result['lr'], result['reg'],
                                                                                   result['factor'],
                                                                                   result['min_val_loss']))
    return result
