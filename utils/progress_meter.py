import sys

import torch
from tqdm import tqdm

__all__ = ['progressMeter']


class progressMeter:
    def __init__(self, desc, writer, logger, total_data_len, epoch, use_pbar):
        self.states = torch.zeros(1, dtype=torch.float32)
        self.states_name_idx = {'batch_size': 0}
        self.writer = writer
        self.logger = logger
        self.epoch = epoch
        self.desc = f'Epoch {epoch} | {desc}: '
        self.total_data_len = total_data_len
        self.pbar = tqdm(total=total_data_len, dynamic_ncols=True, file=sys.stdout, desc=self.desc) if use_pbar else None

    def update(self, state_dict):
        assert 'batch_size' in state_dict, "Batch size should be provided in states! (as 'batch_size')"
        assert 'loss' in state_dict, "Loss should be provided in states! (as 'loss')"
        with torch.no_grad():
            self.state_size = state_dict['batch_size']
            self.states[0] += self.state_size
            for name, val in state_dict.items():
                if name == 'batch_size':
                    continue
                if name not in self.states_name_idx:
                    self.states_name_idx[name] = len(self.states_name_idx)
                    self.states = torch.cat((self.states, torch.zeros(1)))
                if torch.is_tensor(val):
                    val = val.detach().cpu()
                self.states[self.states_name_idx[name]] += val * self.state_size

    def _averages(self):
        return {name: self.states[idx] / self.states[0]
                for name, idx in self.states_name_idx.items() if name != 'batch_size'}

    def verbose_states(self):
        averages = self._averages()
        if self.pbar is not None:
            self.pbar.set_postfix({name: f'{value:.5f}' for name, value in averages.items()}, refresh=False)
            self.pbar.update(int(self.state_size))
        else:
            print(self.desc + ''.join(f' | {name}: {value:.5f}' for name, value in averages.items()))

    def write_summary(self):
        averages = self._averages()
        for name, value in averages.items():
            self.writer.add_scalar(name, value, self.epoch)
        out = ''.join(f' | {name}: {value:.5f}' for name, value in averages.items())
        epoch_prog = f'({self.states[0]:4.0f}/{self.total_data_len:4.0f})'
        self.logger.debug(f'Epoch Summary | {self.desc}{epoch_prog}{out}')
        if self.pbar is not None:
            self.pbar.close()
        return averages['loss']
