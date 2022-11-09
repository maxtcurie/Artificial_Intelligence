Way to install

https://pytorch.org/get-started/locally/#no-cuda
Type: conda install pytorch torchvision torchaudio cpuonly -c pytorch



One migh encounter the warning

Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
                 It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe
Traceback (most recent call last):
  File "Learn_draw_numbers.py", line 1, in <module>
    import torch
  File "C:\ProgramData\Anaconda3\lib\site-packages\torch\__init__.py", line 128, in <module>
    raise err
OSError: [WinError 126] The specified module could not be found. Error loading "C:\ProgramData\Anaconda3\lib\site-packages\torch\lib\asmjit.dll" or one of its dependencies.


One needs to download from https://aka.ms/vs/16/release/vc_redist.x64.exe