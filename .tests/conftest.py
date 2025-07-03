import sys
import types

# Create stubs for open_webui modules used by the pipeline
open_webui = types.ModuleType("open_webui")
models_mod = types.ModuleType("open_webui.models")
chats_mod = types.ModuleType("open_webui.models.chats")

class Chats:
    @staticmethod
    def get_chat_by_id(chat_id):
        return None

class ChatModel:
    def __init__(self, chat=None):
        self.chat = chat or {}

chats_mod.Chats = Chats
chats_mod.ChatModel = ChatModel

models_models_mod = types.ModuleType("open_webui.models.models")
class Models:
    @staticmethod
    def get_model_by_id(model_id):
        return None

    @staticmethod
    def update_model_by_id(model_id, model_form):
        return False

class ModelForm:
    def __init__(self, **kwargs):
        pass

class ModelParams:
    def __init__(self, **kwargs):
        pass

class Model:
    def __init__(self, **kwargs):
        pass

class ModelMeta:
    def __init__(self, **kwargs):
        pass

models_models_mod.Models = Models
models_models_mod.ModelForm = ModelForm
models_models_mod.ModelParams = ModelParams
models_models_mod.Model = Model
models_models_mod.ModelMeta = ModelMeta

utils_mod = types.ModuleType("open_webui.utils")
misc_mod = types.ModuleType("open_webui.utils.misc")

def get_message_list(*args, **kwargs):
    return []

def get_system_message(messages):
    for msg in messages or []:
        if msg.get("role") == "system":
            return msg
    return None

misc_mod.get_message_list = get_message_list
misc_mod.get_system_message = get_system_message
utils_mod.misc = misc_mod

tasks_mod = types.ModuleType("open_webui.tasks")

def list_task_ids_by_chat_id(chat_id):
    return []

async def stop_task(task_id):
    return True

tasks_mod.list_task_ids_by_chat_id = list_task_ids_by_chat_id
tasks_mod.stop_task = stop_task

internal_db_mod = types.ModuleType("open_webui.internal.db")

class Base:
    pass

class JSONField:
    pass

def get_db():
    return None

internal_db_mod.Base = Base
internal_db_mod.JSONField = JSONField
internal_db_mod.get_db = get_db

sys.modules.setdefault("open_webui", open_webui)
sys.modules.setdefault("open_webui.models", models_mod)
sys.modules.setdefault("open_webui.models.chats", chats_mod)
sys.modules.setdefault("open_webui.models.models", models_models_mod)
sys.modules.setdefault("open_webui.utils", utils_mod)
sys.modules.setdefault("open_webui.utils.misc", misc_mod)
sys.modules.setdefault("open_webui.tasks", tasks_mod)
sys.modules.setdefault("open_webui.internal.db", internal_db_mod)
