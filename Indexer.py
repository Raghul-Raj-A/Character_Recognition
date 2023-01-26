def matrix(num):
    height = 5 # gets from the main code 
    width = 5
    row = 1
    col= num
    if num > height * width:
        print("Available matrix is in : {} * {} \n Total number of elements : {}".format(height,width,(height*width)))    
    else:
        while col > width:
            col  = col - width
            row = row+1
        return row,col

print(matrix(int(input("Enter element : "))))

'''

a,b = matrix(int(input("Enter element : ")))
print("Row : {} \nCol : {} ".format(a,b))

'''