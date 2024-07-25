class AutoStr:
    def __str__(self):
        # 获取当前对象的所有属性和值
        attributes = ", ".join(f"{attr}={getattr(self, attr)}" for attr in dir(self) if
                               not callable(getattr(self, attr)) and not attr.startswith("__"))
        return f"{self.__class__.__name__}({attributes})"


# this is the code file for different types of message and their response
class Message_ready(AutoStr):
    def __init__(self):
        self.type = 'ready'

class Message_ready_response(AutoStr):
    def __init__(self, ready=False):
        self.type = 'ready_response'
        self.ready = ready

class Message_close_server(AutoStr):
    def __init__(self, server_id):
        self.type = 'close_server'
        self.server_id = server_id


class Message_close_response(AutoStr):
    def __init__(self, server_id, count):
        self.type = 'close_server_response'
        self.server_id = server_id
        self.count = count


class Message_connect(AutoStr):
    def __init__(self, server_id):
        self.type = 'connect'
        self.server_id = server_id


class Message_Query(AutoStr):
    def __init__(self, data_id, request_type=0):
        self.type = 'data_query'
        self.data_id = data_id
        self.request_type = request_type


class Message_Query_Response(AutoStr):
    def __init__(self, data_id, data, status, trans_hop):
        self.type = 'data_query_response'
        self.data_id = data_id
        self.data = data
        self.status = status
        self.trans_hop = trans_hop


class Message_data_cache(AutoStr):
    def __init__(self, server_id, data):
        self.type = 'data_cache'
        self.server_id = server_id
        self.data = data


class Message_hbfc_update(AutoStr):
    def __init__(self, server_id, data_id, status):
        self.type = 'hbfc_update'
        self.server_id = server_id
        self.data_id = data_id
        self.status = status  # False for delete; True for insert


class Message_Data_Heat(AutoStr):
    def __init__(self, server_id, data_id):
        self.type = 'data_heat'
        self.server_id = server_id
        self.data_id = data_id


class Message_Data_Heat_Response(AutoStr):
    def __init__(self, server_id, data_id, heat, cover=None, need=False):
        self.type = 'data_heat_response'
        self.server_id = server_id
        self.data_id = data_id
        self.heat = heat
        self.cover = cover
        self.need = need


class Message_Data_Deduplicate(AutoStr):
    def __init__(self, server_id):
        self.type = 'data_deduplicate'
        self.server_id = server_id


class Message_data_predelete(AutoStr):
    def __init__(self, server_id, data_id):
        self.type = 'data_predelete'
        self.server_id = server_id
        self.data_id = data_id


class Message_Delete_Permission(AutoStr):
    def __init__(self, server_id, data_id):
        self.type = 'delete_permission'
        self.server_id = server_id
        self.data_id = data_id


class Message_Delete_Permission_Response(AutoStr):
    def __init__(self, server_id, data_id, permission=False):
        self.type = 'delete_permission_response'
        self.server_id = server_id
        self.data_id = data_id
        self.permission = permission

class Message_data_delete_cancel(AutoStr):
    def __init__(self, server_id, data_id):
        self.type = 'data_delete_cancel'
        self.server_id = server_id
        self.data_id = data_id


class Message_Data_Check(AutoStr):
    def __init__(self, server_id, data_id):
        self.type = 'data_check'
        self.server_id = server_id
        self.data_id = data_id


class Message_Data_Check_Response(AutoStr):
    def __init__(self, server_id, data_id, safe=False):
        self.type = 'data_check_response'
        self.server_id = server_id
        self.data_id = data_id
        self.safe = safe


class Message_register(AutoStr):
    def __init__(self, server_id, data_id):
        self.type = 'register'
        self.server_id = server_id
        self.data_id = data_id


class Message_register_response(AutoStr):
    def __init__(self, server_id, state=False):
        self.type = 'register_response'
        self.server_id = server_id
        self.state = state
