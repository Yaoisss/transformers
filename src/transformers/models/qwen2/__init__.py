# Copyright 2024 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

# 导入类型检查的标记，用于在运行时检查类型
from typing import TYPE_CHECKING

# 导入自定义的实用函数和类
from ...utils import (
    OptionalDependencyNotAvailable,  # 用于处理可选依赖项不可用的情况
    _LazyModule,  # 用于延迟导入模块直到它们真正需要时
    is_tokenizers_available,  # 检查tokenizers依赖是否可用
    is_torch_available,  # 检查PyTorch依赖是否可用
)

# 定义一个字典，其中包含了不同模块的导入结构
_import_structure = {
    "configuration_qwen2": ["QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Qwen2Config"],  # qwen2配置相关的导入
    "tokenization_qwen2": ["Qwen2Tokenizer"],  # qwen2分词相关的导入
}

# 尝试导入tokenizers依赖，如果不可用，引发异常
try:
    if not is_tokenizers_available():  # 检查tokenizers是否可用
        raise OptionalDependencyNotAvailable()  # 如果不可用，引发异常
except OptionalDependencyNotAvailable:  # 如果引发异常，则忽略并继续
    pass  # 忽略异常
else:
    # 如果tokenizers可用，添加快速分词器的导入结构
    _import_structure["tokenization_qwen2_fast"] = ["Qwen2TokenizerFast"]

# 尝试导入PyTorch依赖，如果不可用，引发异常
try:
    if not is_torch_available():  # 检查PyTorch是否可用
        raise OptionalDependencyNotAvailable()  # 如果不可用，引发异常
except OptionalDependencyNotAvailable:  # 如果引发异常，则忽略并继续
    pass  # 忽略异常
else:
    # 如果PyTorch可用，添加模型相关的导入结构
    _import_structure["modeling_qwen2"] = [
        "Qwen2ForCausalLM",
        "Qwen2Model",
        "Qwen2PreTrainedModel",
        "Qwen2ForSequenceClassification",
    ]

# 如果处于类型检查状态，导入相应的模块
if TYPE_CHECKING:
    from .configuration_qwen2 import QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP, Qwen2Config  # 导入qwen2配置相关的模块
    from .tokenization_qwen2 import Qwen2Tokenizer  # 导入qwen2分词相关的模块

    # 尝试导入tokenizers依赖，如果不可用，引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果tokenizers可用，导入快速分词器模块
        from .tokenization_qwen2_fast import Qwen2TokenizerFast

    # 尝试导入PyTorch依赖，如果不可用，引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果PyTorch可用，导入模型相关的模块
        from .modeling_qwen2 import (
            Qwen2ForCausalLM,
            Qwen2ForSequenceClassification,
            Qwen2Model,
            Qwen2PreTrainedModel,
        )

# 在非类型检查环境下（实际运行代码时）
else:
    import sys  # 导入sys模块
    #  使用 LazyModule 动态加载各个子模块中的类，根据当前环境判断是否可以导入 torch 和 tokenizers 相关内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)