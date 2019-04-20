score = [88, 71, 56, 79, 95, 78, 65, 91, 85, 99];
grade = [];
for i in score:
    if i >= 90 and i <= 100:
        grade.append('A');
    elif i >= 80 and i <= 89:
        grade.append('B');
    elif i >= 70 and i <= 79:
        grade.append('C');
    else:
        grade.append('D');

print(grade);
