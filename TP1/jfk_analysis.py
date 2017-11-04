with open('data.csv') as f:
    read_data = f.read()
    lol = read_data.splitlines()
    fields = len(lol[0].split(";"))
    total_lines = len(lol)-1
    counter = 0
    splitlines = []

    for line in lol[1:]:
        splitlines.append(line.split(';'))
        if len(line.split(';')) == fields:
            continue
        else:
            counter += 1

    print("There are " + str(total_lines) + " lines")
    if counter == 0 : print("All lines have " + str(fields) + " fields")
    else : print("There are " + str(counter) +" lines that miss or have too much data")

    noPageCount = 0
    PageCount = []
    avgPageCount = 0

    for line in splitlines:
        if line[11] != "" and line[11].isdigit():
            PageCount.append(int(line[11], 10))
            if int(line[11], 10) == 0: print("This document has 0 pages \n" + str(line) + "\n It is explained in the pdf online that the serial is missing")
        else:
            noPageCount += 1

    avgPageCount = sum(PageCount)/len(PageCount)
    maxPage = max(PageCount)
    minPage = min(PageCount)

    documentTypes = []
    agencies = []
    documentsAgencies = []
    dicType = {}
    dicAgency = {}

    for row in splitlines:

        if row[6] in dicType:
            dicType[row[6]] += 1

        if row[6] not in dicType:
            dicType[row[6]] = 1

        if row[4] in dicAgency:
            dicAgency[row[4]] += 1

        if row[4] not in dicAgency:
            dicAgency[row[4]] = 1

    print('number of documents types : ', len(dicType.keys()))
    print('number of agencies : ', len(dicAgency.keys()))

    for agency in dicAgency.items():

        print(agency[0] + " : " + str(agency[1]))

    f.closed