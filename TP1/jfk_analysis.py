with open('data.csv') as f:
    read_data = f.read()
    lol = read_data.splitlines()
    fields = len(lol[0].split(";"))
    total_lines = len(lol)-1
    counter = 0
    for line in lol[1:]:
        if len(line.split(';')) == fields:
            continue
        else:
            counter += 1

    print("There are " + str(total_lines) + " lines")
    if counter == 0 : print("All lines have " + str(fields) + " fields")
    else : print("There are " + str(counter) +" lines that miss or have too much data")
    f.closed