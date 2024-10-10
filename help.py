import re
import time
# how to handle struct generate

cfunc = {
    "return_type": None,
    "function_name": None,
    "arg_type": []
}

arg = {
    "arg_type": None,
    "arg_name": None
}

def remove_more_space(string):
    return re.sub(r"\s+", " ", string)


def parse_c_func(c_func_string):
    # c_func_string = "untensor find_optimal(untensor tensor);"
    c_func_string = remove_more_space(c_func_string)
    head = c_func_string.split("(")[0].strip()
    tail = c_func_string.split("(")[1].split(")")[0].strip()
    
    function_name = head.split(" ")[-1]
    return_type   = head[:-len(function_name)-1].strip()
    argslist = tail.split(",")
    args = []
    for arg in argslist:
        arg      = arg.strip()
        arg_name = arg.split(" ")[-1].strip()
        arg_type = arg[:-len(arg_name)-1].strip()
        args.append({"arg_type": arg_type, "arg_name": arg_name})
        
    return {"return_type": return_type, "function_name": function_name, "arg_type": args}

type_map = {
    "bool"           : "ctypes.c_bool",
    "char"           : "ctypes.c_char",
    "wchar_t"        : "ctypes.c_wchar",
    "unsinged char"  : "ctypes.c_ubyte",
    "short"          : "ctypes.c_short",
    "unsigned short" : "ctypes.c_ushort",
    
    "int"            : "ctypes.c_int",
    "uint"           : "ctypes.c_uint",
    "uint8_t"        : "ctypes.c_uint8",
    "uint32_t"       : "ctypes.c_uint32",
    "int32_t"        : "ctypes.c_int32",
    "uint64_t"       : "ctypes.c_uint64",
    "long"           : "ctypes.c_long",
    "long long"      : "ctypes.c_longlong",
    "unsigned long"  : "ctypes.c_ulonglong",
    "size_t"         : "ctypes.c_size_t",
    "ssize_t"        : "ctypes.c_ssize_t",
    "time_t"         : "ctypes.c_time_t",
    
    "float"          : "ctypes.c_float",
    "double"         : "ctypes.c_double",
    "char*"          : "ctypes.c_char_p",
    "void*"          : "ctypes.c_void_p",
    "float*"         : "ctypes.c_void_p",
    "int*"           : "ctypes.c_void_p",
}

user_type_map = {
    "u64"                   : "ctypes.c_uint64",
    "u32"                   : "ctypes.c_uint32",
    "struct un_runtime_s*"  : "ctypes.c_void_p",
    "uint64_t"              : "ctypes.c_uint64",
    "untensor"              : "ctypes.POINTER(UntensorS)",
    "untensor_s*"           : "ctypes.POINTER(UntensorS)",
    "struct model_info_s_c*": "ctypes.POINTER(ModelInfoSC)",
    "struct model_info_s*"  : "ctypes.c_void_p",
    "bm_handle_t"           : "ctypes.c_void_p",
    "ModelCtx*"             : "ctypes.c_void_p",
    # bmrt 
    "struct model_info_s*"  : "ctypes.c_void_p",
    "Binary*"         : "ctypes.c_void_p",# need fix 
    "struct api_info_t*"    : "ctypes.c_void_p",
}

def_args_map = {
    "ctypes.POINTER(UntensorS)"         : "ctypes.byref({0})",
    "ctypes.POINTER(ctypes.c_uint64)"   : "make2_c_uint64_list({0})",
    "ctypes.POINTER(ctypes.c_int)"      : "make2_c_int_list({0})",
    "ctypes.c_char_p"                   : "str2char_point({0})",
    "ctypes.POINTER(ModelInfoSC)"       : "ctypes.byref({0})",
    "ctypes.c_uint64"                   : "ctypes.c_uint64({0})",
    "ctypes.c_uint32"                   : "ctypes.c_uint32({0})",
    "ctypes.c_int"                      : "ctypes.c_int({0})",
    "ctypes.c_long"                     : "ctypes.c_long({0})",
    "ctypes.c_longlong"                 : "ctypes.c_longlong({0})",
    "ctypes.c_ulonglong"                : "ctypes.c_ulonglong({0})",
    "ctypes.c_size_t"                   : "ctypes.c_size_t({0})",
    "ctypes.c_ssize_t"                  : "ctypes.c_ssize_t({0})"
}

struct_map = {}
total_map = {}

def update_map(user_map_file=None):
    global total_map
    total_map.update(type_map)
    total_map.update(user_type_map)
    if user_map_file is None:
        pass
    else:
        # key,value
        fcontent = open(user_map_file,'r').read().split("\n")
        for line in fcontent:
            if line.strip() == "":
                continue
            key = line.split(",")[0].strip()
            value = line.split(",")[1].strip()
            total_map[key] = value
    return total_map

def convert_type_into_ctypes_with_pointer(type_name, type_map):
    if "*" not in type_name:
        return None
    type_name_without_pointer = type_name.replace("*", "").strip()
    if type_name_without_pointer not in type_map:
        return "ctypes.c_void_p"
        # raise Exception("type_name: {} not in type_map".format(type_name))
    else:
        return "ctypes.POINTER"+"("+type_map[type_name_without_pointer]+")"

def convert_type_into_ctypes(type_name, type_map):
    if "const" in type_name:
        type_name = type_name.replace("const", "").strip()
    if " *" in type_name:
        type_name = type_name.replace(" *", "*").replace(" *", "*").strip()
    if type_name not in type_map:
        if "*" in type_name:
            return convert_type_into_ctypes_with_pointer(type_name, type_map)
        raise Exception("type_name: {} not in type_map".format(type_name))
    else:
        return type_map[type_name]
    return convert_type_into_ctypes(type_name, type_map)

def convert_into_ctypes(data_type_dict, type_map):
    argsres = []
    restype = None
    return_type = data_type_dict["return_type"]
    
    if not return_type == "void":
        restype = convert_type_into_ctypes(return_type, type_map)
    else:
        restype = "None"
    data_type_dict["return_type"] = restype
    function_name = data_type_dict["function_name"]
    args = data_type_dict["arg_type"]
    for arg in args:
        arg_type = arg["arg_type"]
        arg_name = arg["arg_name"]
        type_name = convert_type_into_ctypes(arg_type, type_map)
        argsres.append( type_name )  
    
    return restype, function_name, argsres

def convert_into_str(restype, function_name, argsres):
    retres = None
    argres = None
    res = ""
    if restype:
        retres = f"lib.{function_name}.restype  = {restype}"
        res += retres + "\n"
    if argsres:
        argsres = "[" + ", ".join(argsres) + "]"
        argres = f"lib.{function_name}.argtypes = {argsres}"
        res += argres + "\n"
    return res

def convert_func_string_into_def(parse_dict, argsres, source_line):
    function_name = parse_dict["function_name"]
    args_name     = [i['arg_name'] for i in parse_dict["arg_type"]]
    res = f"def {function_name}("
    res += ", ".join(args_name)
    res += ") -> {}:".format(parse_dict["return_type"])
    comment   = '    """\n'
    comment  += '    ' + remove_more_space(source_line).strip() + "\n"
    for idx in range(len(args_name)):
        idx_type = argsres[idx]
        idx_name = args_name[idx]
        comment += "    :param " + idx_name + ": \t" + idx_type + "\n"
    comment += '    """\n'
    func_body = "    return lib." + function_name + "("
    each_res = []
    for idx in range(len(args_name)):
        idx_type = argsres[idx]
        idx_name = args_name[idx]
        if idx_type in def_args_map:
            each_res.append(def_args_map[idx_type].format(idx_name))
        else:
            each_res.append(idx_name)
    func_body += ", ".join(each_res)
    func_body += ")"
    return res + "\n" + comment + func_body+"\n"

def convert_func_string_into_ctype_string(func_string, type_map):
    if "(" not in func_string or ")" not in func_string:
        return ""
    parse_dict = parse_c_func(func_string)
    restype, function_name, argsres = convert_into_ctypes(parse_dict, type_map)
    res = convert_into_str(restype, function_name, argsres)
    res += convert_func_string_into_def(parse_dict, argsres,func_string)
    return res

def make_struct_name_up(struct_name):
    item = struct_name.split("_")
    # each item first letter up
    item = [i[0].upper()+i[1:] for i in item]
    return "".join(item)

def convert_type_into_ctypes_with_struct(type_name, type_map):
    global struct_map
    if type_name in struct_map:
        return struct_map[type_name]
    return convert_type_into_ctypes(type_name, type_map)

def parse_struct_str(struct_str, type_map):
    global struct_map
    struct_str = struct_str.replace("\n","")
    struct_str = remove_more_space(struct_str)
    # struct struct_name { args };
    # args: type arg_name;
    head       = struct_str.split("{")[0].strip()
    tail       = struct_str.split("{")[1].split("}")[0].strip()
    struce_name = head.split(" ")[-1].strip()
    argslist    = tail.split(";")
    struct_name_up = make_struct_name_up(struce_name)
    struct_map[struce_name] = struct_name_up
    new_args    = []
    for arg in argslist:
        arg = arg.strip()
        if arg == "":
            continue
        arg_name = arg.split(" ")[-1]
        arg_type = arg[:-len(arg_name)-1].strip()
        new_arg_type = convert_type_into_ctypes_with_struct(arg_type, type_map)
        if arg_name.endswith("]") and "[" in arg_name:
            num = arg_name.split("[")[1][:-1]
            arg_name = arg_name.split("[")[0]
            new_arg_type = new_arg_type + " * " + num
        new_args.append({"arg_type": new_arg_type, "arg_name": arg_name})
    return struct_name_up, new_args

def convert_struct_into_ctype_with_self_pointer(struct_name_up, new_args):
    res = "class " + struct_name_up + "(ctypes.Structure):\n"
    # add pass
    res += "    pass\n"
    res += struct_name_up+"._fields_ = [\n"
    steps = []
    for  idx, arg in enumerate(new_args):
        if struct_name_up in arg["arg_type"]:
            steps.append(idx)
            res += "        ('" + arg["arg_name"] + "', " + "ctypes.c_void_p" + "),\n"
            continue
        res += "        ('" + arg["arg_name"] + "', " + arg["arg_type"] + "),\n"
    res += "    ]\n"
    for idx in steps:
        res += struct_name_up+"._fields_[" + str(idx) + "] = " + "('" + new_args[idx]["arg_name"] + "', " + new_args[idx]["arg_type"] + ")\n"
    return res

def convert_struct_into_ctype(struct_str, type_map):
    global total_map
    struct_name_up, new_args = parse_struct_str(struct_str, type_map)
    for arg in new_args:
        if struct_name_up in arg["arg_type"]:
            return convert_struct_into_ctype_with_self_pointer(struct_name_up, new_args)
    res = "class " + struct_name_up + "(ctypes.Structure):\n"
    res += "    _fields_ = [\n"
    for arg in new_args:
        res += "        ('" + arg["arg_name"] + "', " + arg["arg_type"] + "),\n"
    res += "    ]\n\n"
    total_map.update(struct_map)
    struct_name_up_type = "ctypes.POINTER({})".format(struct_name_up)
    def_args_map[struct_name_up_type] = "ctypes.byref({0})"
    return res

# timestamp 
def add_comment():
    day_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    return "version='"+day_time+"'"+"\n"

def refine_func_string(func_string):
    # each items split by \n\n
    func_string = func_string.replace("\n\n", "|split|")
    func_string = func_string.replace("\n", " ")
    func_string = func_string.replace("|split|", "\n")
    return func_string

def parse_generate(func_path, type_map_path=None, output_path="./generate.txt", template_path=None):
    fcontent = open(func_path, 'r').read()
    fcontent = refine_func_string(fcontent)
    fcontent = fcontent.split("\n")
    res = ""
    type_map = update_map(type_map_path)
    temp = ""
    is_end = True
    for line in fcontent:
        if line.strip() == "":
            continue
        if line.strip().startswith("struct") and "{" in line and "}" not in line:
            is_end = False
            temp += line + "\n"
            continue
        if not is_end:
            temp += line + "\n"
            if "};" in line:
                res += convert_struct_into_ctype(temp, type_map)
                temp = ""
                is_end = True
            continue
        # res += "\n#" + remove_more_space(line).strip()
        res += "\n"
        res += convert_func_string_into_ctype_string(line, type_map)
    res += "\n" + add_comment()
    print(res)
    tres = open(template_path, 'r').read()
    
    open(output_path, 'w').write(tres + res)

if __name__=="__main__":
    parse_generate("./func.txt", None, "./untool.py", "./template.py")