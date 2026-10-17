# About this project

Programming language: python
A small example which uses mujoco to simulate a hugging face reachy mini robot.
The robot will use a local llm (running in ollama) over the openai api to reason about his actions. 

# rules

## general

- Keep track of all decisions made in the chat this project in @./decisions.md

## Python

- Use the anaconda python executable in `C:\Users\lbsupan\AppData\Local\anaconda3\python.exe` as default.
- Never install any python package in the base environment, neither with pip nor with conda!
- use venv to create local environments
- write a requirements.txt file for everything that has been installed so that the requirement can easily be recreated.

# run the project

`.\reachy_test_14\Scripts\python.exe run_sim_daemon.py` terminal 1: simulator
`.\reachy_test_14\Scripts\python.exe llmtest.py` terminal 2: application

`.\reachy_test_14\Scripts\python.exe voicechat.py` terminal 2 (alternative): voice chat demo
(Whisper -> local LLM -> Kokoro-82M TTS -> Reachy Mini speakers). Requires the
Kokoro-82M ONNX model files in `tts_models/` (not in git, see decisions.md for the
download links): `tts_models/kokoro-v1.0.onnx` and `tts_models/voices-v1.0.bin`.


