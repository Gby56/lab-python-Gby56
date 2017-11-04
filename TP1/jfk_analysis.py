import datetime

with open('data.csv') as f:
    read_data = f.read()
    lol = read_data.splitlines()
    fields = len(lol[0].split(";"))
    total_lines = len(lol)-1
    counter = 0
    splitlines = []

    for line in lol[1:]:
        if len(line.split(';')) == fields:
            splitlines.append(line.split(';'))
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
    print(avgPageCount)
    print(maxPage)
    print(minPage)
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

    print('number of documents per agency')

    for agency in dicAgency.items():

        print(agency[0] + " : " + str(agency[1]))

    dates = []
    unsupportedformats = 0
    dicYears = {}

    for idx, row in enumerate(splitlines):
        try:
            date = datetime.datetime.strptime(row[5],'%m/%d/%Y')
            dates.append(date)

            if date.year not in dicYears:
                dicYears[date.year] = 1

            if date.year in dicYears:
                dicYears[date.year] += 1


        except ValueError:
            unsupportedformats+=1

    print("There are %d documents with unsupported dates" % (unsupportedformats))
    print("The oldest document is from %s" % (min(dates)))
    print("The most recent document is from %s" % (max(dates)))

    for year in dicYears.items():
        print(str(year[0]) + " : " + str(year[1]))

    f.closed