score = [88, 71, 56, 79, 95, 78, 65, 91, 85, 99];
grade = [];
i = 0;
while i < 10:
    if score[i] >= 90 and score[i] <= 100:
        grade.append('A');
    elif score[i] >= 80 and score[i] <= 89:
        grade.append('B');
    elif score[i] >= 70 and score[i] <= 79:
        grade.append('C');
    else:
        grade.append('D');
    i = i + 1;

print(grade);
