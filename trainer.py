from copy import deepcopy
from datetime import datetime
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class MyEngine(Engine):

    def __init__(self, func, model, crit, optimizer, config):

        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func)

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, mini_batch):

        engine.model.train()
        engine.optimizer.zero_grad()

        x, y = mini_batch

        x, y = x.to(engine.device), y.to(engine.device)

        y_hat = engine.model(x)

        loss = engine.crit(y_hat, x)
        loss.backward()

        engine.optimizer.step()

        return {
            'loss':float(loss),
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch
            x, y = x.to(engine.device), y.to(engine.device)

            y_hat = engine.model(x)

            loss = engine.crit(y_hat, x)

        return {
            'loss':float(loss),
        }

    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):

        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x : x[metric_name]).attach(
                engine,
                metric_name,
            )

        training_metric_names = ['loss']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - loss={:.4e}'.format(
                    engine.state.epoch,
                    engine.state.metrics['loss'],
                ))

        validation_metric_names = ['loss']

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss={:.4e} best_loss={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.best_loss,
                ))

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model':engine.best_model,
                'config':config,
                **kwargs
            },'./CheckPoint/' + str(datetime.now())[:10] + ' ' + config.model_fn
        )


class Trainer():

    def __init__(self, config):
        self.config = config

    def train(self,model, crit, optimizer,train_loader, valid_loader):

        train_engine = MyEngine(
            MyEngine.train,
            model, crit, optimizer, self.config
        )

        validation_engine = MyEngine(
            MyEngine.validate,
            model, crit, optimizer, self.config
        )

        MyEngine.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine, valid_loader,
        )

        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            MyEngine.check_best,
        )

        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            MyEngine.save_model,
            train_engine, self.config,
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs
        )

        return model
