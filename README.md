# fast-chatglm
Fast ChatGLM-6B with CTranslate2

Warning: The project is currently under development, and the decoding results are poor. For more details, please refer to the following link. 
https://github.com/OpenNMT/CTranslate2/issues/1202

## Quick Start
Install requiremnts
```
pip install -r requiremnts.txt
```
Convert ChatGLM-6B model to CTranslate2 format
```
python fast_chatglm/convert_chatglm6b --output_dir chatglm --force --quantization int8
```
Run simple generation example
```
python python fast_chatglm/cli.py chatglm/
```
