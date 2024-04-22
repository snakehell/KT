import numpy as np
import math


class Mesh:
    def __init__(self, filepath):
        self.vertices = self.loadMesh(filepath)
        self.num_objects = len(self.vertices)
        print("Number of objects", self.num_objects, ", number of faces", [len(x)/9 for x in self.vertices])
        self.s_i, self.s_ij = self.countArea()
        
    def loadMesh(self, filepath):
        vertices = []
        obj_vertices = []
        v = []
        index = []
        
        with open(filepath, "r") as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                
                if flag == "v":
                    line = line.replace("v ", "").strip()
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                elif flag == "f":
                    line = line.replace("f ", "").strip()
                    line = line.split(" ")
                    faceVertices = []
                    for vert in line:
                        faceVertices.append(v[int(vert) - 1])
                    for i in [0,1,2]:
                        for x in faceVertices[i]:
                            obj_vertices.append(x)
                elif flag == "g":
                    index.append(line[-2])
                    if len(obj_vertices):
                        vertices.append(obj_vertices)
                    obj_vertices = []
                else:
                    pass
                
                line = f.readline()
        vertices.append(obj_vertices)
        vertices = [x for _, x in sorted(zip(index, vertices), key=lambda pair: pair[0])]
        return vertices
    
    def countArea(self):
        S_I = np.zeros((self.num_objects))
        S_IJ = np.zeros((self.num_objects, self.num_objects))
            
        # S[0] = pi*(r1 + r2)*l + pi*r1^2
        z_max = max([self.vertices[0][x] for x in range(len(self.vertices[0])) if x%3 == 2]) 
        z_min = min([self.vertices[0][x] for x in range(len(self.vertices[0])) if x%3 == 2])
        r_big = max([self.vertices[0][x] for x in range(len(self.vertices[0])) if x%3 == 0])
        r_small = max([self.vertices[0][x] for x in range(len(self.vertices[0]) - 2) if self.vertices[0][x+2] == z_max and x%3 == 0])
        l = math.sqrt((z_max - z_min)**2 + (r_big - r_small)**2)
        
        S_I[0] = math.pi*((r_big + r_small)*l + r_small**2)
        
        # S[1] =  2*pi*r*h
        r = max([self.vertices[1][x] for x in range(len(self.vertices[1])) if x%3 == 0])
        z_max = max([self.vertices[1][x] for x in range(len(self.vertices[1])) if x%3 == 2]) 
        z_min = min([self.vertices[1][x] for x in range(len(self.vertices[1])) if x%3 == 2])
        h = z_max - z_min
        
        S_I[1] = 2*math.pi*r*h
        S_IJ[1][2] = S_IJ[2][1] = math.pi*r**2
        S_IJ[1][0] = S_IJ[0][1] = math.pi*r**2
        
        # S[3] = 2*pi*r*h
        x_max = max([self.vertices[3][x] for x in range(len(self.vertices[3])) if x%3 == 0]) 
        x_min = min([self.vertices[3][x] for x in range(len(self.vertices[3])) if x%3 == 0])
        y_max = max([self.vertices[3][x] for x in range(len(self.vertices[3])) if x%3 == 1]) 
        y_min = min([self.vertices[3][x] for x in range(len(self.vertices[3])) if x%3 == 1])
        r = abs((x_max - x_min)/2)
        h = abs(y_max - y_min)
        
        S_I[3] = 2*math.pi*r*h
        S_IJ[3][2] = S_IJ[2][3] = math.pi*r**2
        S_IJ[3][4] = S_IJ[4][3] = math.pi*r**2
        
        # S[2] = 2(ab + bc + ac)
        x_max = max([self.vertices[2][x] for x in range(len(self.vertices[2])) if x%3 == 0]) 
        x_min = min([self.vertices[2][x] for x in range(len(self.vertices[2])) if x%3 == 0])
        y_max = max([self.vertices[2][x] for x in range(len(self.vertices[2])) if x%3 == 1]) 
        y_min = min([self.vertices[2][x] for x in range(len(self.vertices[2])) if x%3 == 1])
        z_max = max([self.vertices[2][x] for x in range(len(self.vertices[2])) if x%3 == 2]) 
        z_min = min([self.vertices[2][x] for x in range(len(self.vertices[2])) if x%3 == 2])
        a = abs(x_max - x_min)
        b = abs(y_max - y_min)
        c = abs(z_max - z_min)
        S_I[2] = 2*(a*b + b*c + a*c) - S_I[1] - S_I[3]
        
        # S[4] 
        x_max = max([self.vertices[4][x] for x in range(len(self.vertices[4])) if x%3 == 0]) 
        x_min = min([self.vertices[4][x] for x in range(len(self.vertices[4])) if x%3 == 0])
        y_max = max([self.vertices[4][x] for x in range(len(self.vertices[4])) if x%3 == 1]) 
        y_min = min([self.vertices[4][x] for x in range(len(self.vertices[4])) if x%3 == 1])
        z_max = max([self.vertices[4][x] for x in range(len(self.vertices[4])) if x%3 == 2]) 
        z_min = min([self.vertices[4][x] for x in range(len(self.vertices[4])) if x%3 == 2])
        z_mid = max([self.vertices[4][x] for x in range(2, len(self.vertices[4])) if self.vertices[4][x-1] == y_min and x%3 == 2])
        
        S_I[4] = abs((x_max - x_min)*(y_max - y_min)) + abs((y_max - y_min)*(z_max + z_mid - 2*z_min)) + \
            abs((x_max - x_min)*(z_max - z_min)) + abs((x_max - x_min)* (z_mid - z_min)) + \
            abs((x_max - x_min)*math.sqrt((z_max - z_mid)**2 + (y_max - y_min)**2)) - S_I[3]
        
        print(S_I, "\n", S_IJ)
        
        return S_I, S_IJ
