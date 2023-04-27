# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Config file holding the training configuaration for the trivial eigengame."""

from eigengame import eg_base_config
from eigengame import eg_utils

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Creates and populates the fields of the ConfigDict."""

  config = eg_base_config.get_config()

  config.interval_type = 'steps'
  config.training_steps = 100_000
  config.save_checkpoint_interval = 100
  config.log_train_data_interval = 10

  config.experiment_kwargs.config.epsilon = 0.
  config.experiment_kwargs.config.maximize = False
  config.experiment_kwargs.config.experiment_type = eg_utils.ExperimentType.PLS
  config.experiment_kwargs.config.track_mean = False
  config.experiment_kwargs.config.eigenvector_count = 8

  config.experiment_kwargs.config.optimizer_schedule_config.warm_up_step = 10000
  config.experiment_kwargs.config.optimizer_schedule_config.end_step = 100_000
  config.experiment_kwargs.config.optimizer_schedule_config.base_lr = 25e-5
  config.experiment_kwargs.config.optimizer_schedule_config.end_lr = 25e-6
  config.experiment_kwargs.config.aux_optimizer_config.learning_rate = 25e-4

  config.experiment_kwargs.config.dataset_config = dict(
      global_batch_size=512,
      repeat=True,
      seed=0,
      n_samples=512,
      n_latents=4,
      eigenvector_count=config.experiment_kwargs.config.eigenvector_count,
  )
  config.experiment_kwargs.config.optimizer_schedule_config.end_step = int(1e8)
  config.checkpoint_dir = '/tmp/eigengame_pls_test/'

  return config
