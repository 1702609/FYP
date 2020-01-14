
f1 = [(533,268),(648,352),(1011,403),(972,546),(240,102)]
f2 = [(648,338),(1056,458),(240,102),(972,550),(529,260)]


def distance(x, coordinate):
    x_value = abs(x[0]-coordinate[0])
    y_value = abs(x[1]-coordinate[1])
    return x_value+y_value

newf2 = []
for i in range(len(f2)):
    newf2.append(min(f2, key=lambda x: distance(x, f1[i])))
f2.clear()
f2 = newf2.copy()
newf2.clear()

print(f1)
print(f2)