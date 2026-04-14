import subprocess

print("Executando Sit and Reach...")
subprocess.run(["python", "./Sit-and-Reach/sit_and_reach_holistic_2.py"], check=True)

print("Executando Back Scratch...")
subprocess.run(["python", "./Back-Scratch/back_scratch.py"], check=True)

print("Testes concluídos com sucesso.")