import kornia
import torch

list_kornia_functions = [func for func in dir(kornia) if not func.startswith('__')]

for kornia_func in list_kornia_functions:
    try:
        print("trying to parse : " + kornia_func)

        op = torch.jit.script(getattr(kornia, kornia_func))
        torch.jit.save(op, kornia_func + '.pt')
    except Exception as er:
        print(str(er))
