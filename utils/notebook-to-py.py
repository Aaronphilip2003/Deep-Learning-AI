import sys,json

f = open('./langgraph/Basics/ConditionalAgentsExercise.ipynb', 'r') #input.ipynb
j = json.load(f)
of = open('output.py', 'w') #output.py
if j["nbformat"] >=4:
        for i,cell in enumerate(j["cells"]):
                of.write("#cell "+str(i)+"\n")
                for line in cell["source"]:
                        of.write(line)
                of.write('\n\n')
else:
        for i,cell in enumerate(j["worksheets"][0]["cells"]):
                of.write("#cell "+str(i)+"\n")
                for line in cell["input"]:
                        of.write(line)
                of.write('\n\n')

of.close()