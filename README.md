# fast-chatglm
Faster ChatGLM-6B with CTranslate2


## Quick Start
Install requiremnts
```
pip install -r requiremnts.txt
```
Convert ChatGLM-6B model to CTranslate2 format
```
python faster_chatglm/convert_chatglm6b --output_dir chatglm --force --quantization int8
```
Run simple generation example
```
python python faster_chatglm/cli.py chatglm/
```
