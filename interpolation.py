

def interpolation():
    traj_file = './files/Traj09_part.txt'
    out_file = './files/Traj09_part_interpolation.txt'
    count = 0
    total = 241 - 1
    start = 1.546884999999999977e+01
    end = 6.714588999999999430e+01
    with open(traj_file, 'r') as infile, open(out_file, "w") as outfile:
        for line in infile.readlines():
            line = line.strip()
            pose = line.split(' ')
            pose[0] = str(start + (end - start) * count / total)
            for i in range(8):
                outfile.write(pose[i])
            outfile.write('\n')
            count += 1
            
        infile.close()
        outfile.close()

interpolation()
