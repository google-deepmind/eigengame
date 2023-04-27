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

"""Config file holding the default training configuration for EigenGame."""

from jaxline import base_config
import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Creates and populates the fields of the ConfigDict."""

  config = base_config.get_base_config()

  config.interval_type = 'steps'
  config.training_steps = 10_000_000
  config.save_checkpoint_interval = 10_000
  config.log_train_data_interval = 100

  config.experiment_kwargs = ml_collections.ConfigDict(
      dict(
          config=dict(
              eigenvector_count=128,
              eval_batches=128,
              epsilon=1e-4,
              maximize=True,
              track_mean=True,
              optimizer_schedule_config=dict(
                  warm_up_step=10_000,
                  end_step=1_000_000,
                  base_lr=2e-4,
                  end_lr=1e-6),
              optimizer_config=dict(
                  b1=0.9,
                  b2=0.999,
                  eps=1e-8,
              ),
              aux_optimizer_config=dict(
                  learning_rate=1e-3,
              ),
              dataset_config=dict(),
              preprocess_config=dict(),
          )))
  config.train_checkpoint_all_hosts = True
  return config
