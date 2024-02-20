#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix #compress the memory that a matrix uses
from sklearn.neighbors import NearestNeighbors #knn distance algorithm
import matplotlib.pyplot as plt #plotting graph


# In[3]:


ratings = pd.read_csv("/Users/himangiparekh/Desktop/COMPUTER SCIENCE/CS IA/movieListSmall/ratings.csv")
# imported dataset ratings from computer
# contains information on the ratings given to each movie by each user


# In[4]:


ratings = ratings.rename(columns={'userId': 'userKey', 'movieId': 'movieKey', 'rating': 'ratingGiven', 'timestamp': 'duration'})
# renaming the columns


# In[5]:


ratings.head()
# printing the first five rows of the dataset


# In[10]:


movies = pd.read_csv("/Users/himangiparekh/Desktop/COMPUTER SCIENCE/CS IA/movieListSmall/movies.csv")
# importing movies dataset from computer
# contains information regarding the movieId and movieName + genre
# important to connect the two datasets and user will eventually use movie name not ID
# also will be used for allowing user to subset the recommendations based on genre
movies = movies.rename(columns={'movieId': 'movieKey', 'title': 'nameOfMovie', 'genres': 'genresOfMovie'})
# renaming the columns
movies_list = movies.iloc[:, 2].values
for i in movies_list: 
    movies['nameOfMovie'] = movies['nameOfMovie'].str.replace(':', '')
movies
# printing the first five rows of the dataset


# In[11]:


movies_list = movies.iloc[:, 2].values
for i in movies_list: 
    movies['nameOfMovie'] = movies['nameOfMovie'].str.replace('-', ' ')
movies


# In[9]:


matrix = ratings.pivot(index='movieKey',columns='userKey',values='ratingGiven')
# changing the organisation of the 'ratings' dataset into a matrix:
# with columns as the movieId
# with rows as the userId
# with each cell representing the ratings given by the particular user to the particular
# movie
matrix


# In[12]:


# now we want to reduce the sparsity of the dataset
# we will check to see if each movie has been rated a certain number of times
# if the respective conditions are not satisfied, the row will be erased

non_nan_movies = matrix.count(1)
# storing the number of non 'NAN' in each row (number of ratings given to each movie)
non_nan_movies


# In[13]:


non_nan_movies_list = non_nan_movies.tolist()
# storing the number of ratings given to each movie as a list 
print(non_nan_movies_list)


# In[14]:


figure , axis = plt.subplots(1,1,figsize=(20,5))
# creating a graph/plot
plt.scatter(non_nan_movies.index,non_nan_movies,color='seagreen')
# on a scatter graph, plotting the number of ratings given to each movie
# important to visualise the data and understand its nature (min, max, average etc) 
plt.xlabel('movieKey')
plt.ylabel('Number of ratings given to every movie')
plt.show()


# In[15]:


# we can see that there are several outliers, so setting the benchmark as the average 
# will be of little use
# let us find the range, mode and median

import statistics

print("range =", range(min(non_nan_movies_list), max(non_nan_movies_list)))

def Median_movies(non_nan_movies_list):
    return statistics.median(non_nan_movies_list)

median_movies = Median_movies(non_nan_movies_list)
print("median =", median_movies)

mode_movies = statistics.mode(non_nan_movies_list)
print("mode =", mode_movies)

# since the mode is 1 and the median is 3, there is minimal scope to reduce 
# the sparsity 
# so let us follow the trial and error method to decide the benchmark


# In[16]:


# if statement inside for loop: 
# let us start with 3 as it is the median
# how many indices (which is the number of rows) have values < 3 in list? 

counter =  0 # Counter Variable
number_of_rows = 0
for i in non_nan_movies_list:
    if i < 3:
        #print(counter)
        number_of_rows+=1
    counter+=1 # increments counter
print("Total Rows with less than 3 values: ", number_of_rows)


# In[17]:


# since around half the no. of rows have < 3 values, 
# it does not make sense to remove all these rows, because then the number of movies
# that the user will have access to will be very low
# let us try 2
# how many indices (which is the number of rows) have values < 2 in list? 

counter =  0 # Counter Variable
number_of_rows = 0
for i in non_nan_movies_list:
    if i < 2:
        #print(counter)
        number_of_rows+=1
    counter+=1 # increments counter
print("Total Rows with less than 2 values: ", number_of_rows)


# In[18]:


# once again, it does not make sense to remove 3446 rows out of 9724
# so there is minimal scope to reduce the sparsity of the data

matrix.fillna(0,inplace=True)
# filling NaN values with 0
matrix


# In[2]:


# pd.DataFrame(matrix).to_csv("/Users/himangiparekh/Desktop/COMPUTER SCIENCE/CS IA/mainmatrix.csv")
# pd.DataFrame(movies).to_csv("/Users/himangiparekh/Desktop/COMPUTER SCIENCE/CS IA/mainmovies.csv")
matrix = pd.read_csv("/Users/himangiparekh/Desktop/COMPUTER SCIENCE/CS IA/mainmatrix.csv")

movies = pd.read_csv("/Users/himangiparekh/Desktop/COMPUTER SCIENCE/CS IA/mainmovies.csv")


# In[3]:


compress_data = csr_matrix(matrix.values) # basically used to compress the memory used by data
matrix.reset_index(inplace=True) #creating index column
matrix


# In[4]:


compress_data


# In[5]:


from sklearn.neighbors import NearestNeighbors
# here, we are using the knn algorithm which finds the nearest neighbor/s of a moive
# it plots all the movies in the dataset based on the ratings on a graph
# then, when the user inputs a movie, it will return the 'nearest neighbors' of that movie

knn_algorithm = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# the metric = the way that the algorithm will measure the distance
# cosine is euclidean distance
# n_neighbours is the number of neighbours we want the algorithm the find
# algorithm brute is designed for sparse matrices
# which will be useful for accurately plotting the movies, considering ratings
# and other features (which program will decide)

knn_algorithm.fit(compress_data)
# .fit is a sklearn method which best estimates the representative model of the data
# the function can help the 'metric' part of the algorithm to plot the points most
# efficiently and effectively


# In[6]:


# now we can create a method which will return x no of movies when user inputs a title

def rrecommend_movie(mov_title, number_mov_rec):
    list_of_mov = movies[movies['nameOfMovie'].str.contains(mov_title, case=False)]  # subsets the movies dataset based on user input movie
    if len(list_of_mov): # checking for whether the list is empty or not
        index_mov = list_of_mov.iloc[0]['movieKey']   # finding the movieKey of the 1st element in list
        index_mov = matrix[matrix['movieKey'] == index_mov].index[0]   
        # finding the index from matrix dataframe of the movie key inputted from movies dataset
        knn_distance , knn_index = knn_algorithm.kneighbors(compress_data[index_mov], n_neighbors=number_mov_rec+1)    
        # 2 aspects will be returned when this method is run
        # finding neighbours of the 'movie index' using the knn algorithm and in-built kneighbors method
        # the number of neighbours to find is no. movies to rec + 1 because it also includes the movie that the user inputs (which cannot be considered as a rec)
        mov_rec_index = sorted(list(zip(knn_index.squeeze().tolist(),knn_distance.squeeze().tolist())),key=lambda x: x[1])
        # .squeeze reduces the dimensionality of an array/list
        # zip: making pairs of index and distance
        # creating a tuple with 1st element as index of the movie (movie found by implementing knn algorithm) and second as the distance of that movie from the inputted movie
        # then sorting this tuple based on the second element in each bracket (basically based on the distance) 
        # sorting it based on lamda which is a function to iterate on each element in the list: each variable second element
        # x:x[1] basically means that the sorting is happening in second element of tuple
        arr_recommend = []
        # initialising an empty array which will have recommended movies
        for i in mov_rec_index: # looping through the neighbours we stored as tuple
            index_mov = matrix.iloc[i[0]]['movieKey'] # finding the movie key of the first element of the tuple from the matrix dataset
            index = movies[movies['movieKey'] == index_mov].index # finding the index of the moviekey from the movie database
            arr_recommend.append({'nameOfMovie':movies.iloc[index]['nameOfMovie'].values[0]}) # adding the array with name of movie from the movies dataset of the movie key 
        create_data = pd.DataFrame(arr_recommend,index=range(0,number_mov_rec+1)) # making the array into dataframe and adding an index to it
        #create_data2 = create_data.iloc[1: , :] # removing the first row of create data (as it is the movie itself)
        create_data2 = create_data.iloc[1:, 0]
        return create_data2
    else:
        return "Please input another movie."


# In[7]:


recommend_movie('spider man', 15)


# In[8]:


import tkinter
from tkinter import *
from tkinter import messagebox

import pandas as pd
import pymysql
from PIL import Image, ImageTk    


# In[ ]:


# Login Function

def clear_log():
    un_log.delete(0,END)
    name_log.delete(0, END)
    ps_log.delete(0,END)
    
def clear_sign():
    un_sign.delete(0,END)
    name_sign.delete(0,END)
    ps_sign.delete(0,END)   
    
def exit_log():
    login.destroy()
    
def login_database():

    if un_log.get() == "" or ps_log.get() == "":
        messagebox.showerror("Error!", "Please enter user name and password", parent=login)
    
    else:
        try:
            connect_log = pymysql.connect(host="192.168.64.2", user="root", password="123", database="user_information")
            cursor_log = connect_log.cursor()

            cursor_log.execute("select * from user_information where username=%s and password = %s",(un_log.get(), ps_log.get()))
                                                                                                     
            row = cursor_log.fetchone()
            
            if row == NONE:
                messagebox.showerror("Error", "Invalid User Name And Password", parent=login)

            else:
                messagebox.showinfo(f"Hello {name_log}!",  "You are successfully logged in!", parent=login)
                exit_log()
                homepage()
                
        except Exception as ErrorL:
            return ErrorL
            
# Homepage Panel

def homepage():
    
    def recommend_movie():
        list_of_mov = movies[movies['nameOfMovie'].str.contains(entry_movie_name_home.get(), case=False)]  
        # subsets the movies dataset based on user input movie
        if len(list_of_mov): # checking for whether the list is empty or not
            index_mov = list_of_mov.iloc[0]['movieKey']   # finding the movieKey of the 1st element in list
            index_mov = matrix[matrix['movieKey'] == index_mov].index[0]   
            # finding the index from matrix dataframe of the movie key inputted from movies dataset
            knn_distance , knn_index = knn_algorithm.kneighbors(compress_data[index_mov], n_neighbors=int(entry_number_mov_rec_home.get())+1)
                                                                    
            # 2 aspects will be returned when this method is run
            # finding neighbours of the 'movie index' using the knn algorithm and in-built kneighbors method
            # the number of neighbours to find is no. movies to rec + 1 because it also includes the movie 
            # that the user inputs (which cannot be considered as a rec)
            mov_rec_index = sorted(list(zip(knn_index.squeeze().tolist(),knn_distance.squeeze().tolist())),key=lambda x: x[1])
                                   
            # .squeeze reduces the dimensionality of an array/list
            # zip: making pairs of index and distance
            # creating a tuple with 1st element as index of the movie 
            # (movie found by implementing knn algorithm) and second as the distance of that movie 
            # from the inputted movie)
            # then sorting this tuple based on the second element in each bracket (basically based on the distance) 
            # sorting it based on lamda which is a function to iterate on each element in the list: 
            # each variable second element
            # x:x[1] basically means that the sorting is happening in second element of tuple
            arr_recommend = []
            # initialising an empty array which will have recommended movies
            for i in mov_rec_index: # looping through the neighbours we stored as tuple
                index_mov = matrix.iloc[i[0]]['movieKey'] 
                # finding the movie key of the first element of the tuple from the matrix dataset
                index = movies[movies['movieKey'] == index_mov].index 
                # finding the index of the moviekey from the movie database
                arr_recommend.append({'nameOfMovie':movies.iloc[index]['nameOfMovie'].values[0]}) 
                # adding the array with name of movie from the movies dataset of the movie key 
            create_data = pd.DataFrame(arr_recommend,index=range(0,int(entry_number_mov_rec_home.get())+1)) 
            # making the array into dataframe and adding an index to it
            create_data2 = create_data.iloc[1:, 0]
            tkinter.messagebox.showinfo("Recommendations", ' '.join(create_data2))
        else:
            tkinter.messagebox.showinfo("Error", "Please input another movie.")


    home = Tk()
    home.title("Homepage")
    home.geometry("2000x2000")
    
    mov_name = StringVar()
    mov_number = StringVar()
    
    intro_home = Label(home, text="Welcome to Movie Recommender!", fg='blue', font=("arial", 30, "bold")).pack()

    bag_image = Image.open("/Users/himangiparekh/Desktop/COMPUTER SCIENCE/CS IA/image.png")                         # retrieving a background image from the computer
    background_image = ImageTk.PhotoImage(bag_image)
    label1 = Label(image=background_image)                      # pasting the background image on the login window
    label1.pack(fill=BOTH, expand=TRUE)

    instruction_home = Label(home, text="Please input a movie name along with the number of recommendations that you want to see.", 
                             fg='black', font=("arial", 15, "bold"))
    instruction_home.place(relx=0.27, rely=0.56)

    movie_name_home = Label(home, text="Movie Name : ", font=("arial", 16, "bold"))
    movie_name_home.place(relx=0.34, rely=0.33)

    entry_movie_name_home = Entry(home, textvar=mov_name)
    entry_movie_name_home.place(relx=0.55, rely=0.33)

    number_mov_rec_home = Label(home, text="Number of Movies : ", font=("arial", 16, "bold"))
    number_mov_rec_home.place(relx=0.34, rely=0.4)

    entry_number_mov_rec_home = Entry(home, textvar=mov_number)
    entry_number_mov_rec_home.place(relx=0.55, rely=0.4)

    def test():
        tkinter.messagebox.showinfo("Hello", "How are you")
        
    recommend_mov = Button(home, text="Get Recommendations", width=12, fg='brown', command=recommend_movie)
    recommend_mov.place(relx=0.46, rely=0.675)
    
    def abt_home():                                              
        # initialising a method which shows a message in the home page
        
        tkinter.messagebox.showinfo("About", "Welcome! All you have to do is input a movie which you want your suggestions to be based on! And Voilà!")



    menu_home = Menu(home)                                    # creating menu bar in homepage
    home.config(menu=menu_home)

    About_home = Menu(home)                                         # creating 'about' in menu
    menu_home.add_cascade(label="About", menu=About_home)           # adding dropdown in about
    About_home.add_command(label="What is this program?", command=abt_home)           
    # when clicked, the abt command messagebox will open
    About_home.add_command(label="test", command=recommend_movie)
    home.mainloop()
    
    
# Signup Window


def signup_method():
    un_sign = StringVar()
    name_sign = StringVar()
    ps_sign = StringVar()

    def signup_database():
        
        if entry_un_signup.get() == "" or entry_name_signup.get() =="" or entry_password_signup.get() == "":
            messagebox.showerror("Error!", "Please input all sections.", parent=signup)
            
        else:
            try:
                connect_sign = pymysql.connect(host="192.168.64.2", user="root", password="123", 
                                               database="user_information")
                cursor_sign = connect_sign.cursor()
                cursor_sign.execute("select * from user_information where username=%s", entry_un_signup.get())
                find = cursor_sign.fetchone()
                
                if find!=None:
                    messagebox.showerror("Error!", "User Name Already Exits. Please Select Another User Name.", 
                                         parent=signup)
                
                else:
                    cursor_sign.execute("insert into user_information(username, name, password) values(%s, %s, %s)",
                        (
                        entry_un_signup.get(),
                        entry_name_signup.get(),
                        entry_password_signup.get()
                        ))
                    connect_sign.commit()
                    messagebox.showinfo(f"Hello {entry_name_signup}!", "You are successfully signed up!", 
                                        parent=signup)
                    clear_sign()
                    connect_sign.exit_sign()
                                        
            except Exception as Error:
                return Error
            
    signup = Tk()  # creating window
    signup.title("Sign Up Page")  # giving title
    signup.geometry('2000x2000')  # size

    def exit_sign():
        signup.destroy()
    
    un_signup = Label(signup, text="Username : ", font=("arial", 16, "bold"))
    un_signup.place(relx=0.34, rely=0.33)

    entry_un_signup = Entry(signup, textvar=un_sign)
    entry_un_signup.place(relx=0.55, rely=0.33)

    name_signup = Label(signup, text="Name : ", font=("arial", 16, "bold"))
    name_signup.place(relx=0.34, rely=0.4)

    entry_name_signup = Entry(signup, textvar=name_sign)
    entry_name_signup.place(relx=0.55, rely=0.4)

    password_signup = Label(signup, text="Password : ", font=("arial", 16, "bold"))
    password_signup.place(relx=0.34, rely=0.47)

    entry_password_signup = Entry(signup, textvar=ps_sign, show='*')
    entry_password_signup.place(relx=0.55, rely=0.47)

    intro_signup = Label(signup, text="Welcome to Movie Recommender! Please sign up.", fg='blue', 
                         relief=RAISED, font=("arial", 30, "bold")).pack()

    quit_button_signup = Button(signup, text="Quit", width=12, fg='brown', command=exit_sign)
    quit_button_signup.place(relx=0.53, rely=0.6)

    signup_button_signup = Button(signup, text="Sign Up", width=12, fg='brown', command=signup_database)
    signup_button_signup.place(relx=0.39, rely=0.6)

    login_instruction_signup = Label(signup, text="If you already have an account, please click on 'Login'.", 
                                     fg='black', font=("arial", 14)).place(relx=0.38, rely=0.76)

    login_button_signup = Button(signup, text="Login", width=12, fg='brown', command=signup.destroy)
    login_button_signup.place(relx=0.46, rely=0.675)

    signup.mainloop()
    
# login page

login = Tk()                                # creating a login window
login.geometry("2000x2000")                 # defining the size of the window
login.title("Login Page")                   # giving the window a title

un_log = StringVar()                                   
# username (login page) as a string when user enters it in entry box
name_log = StringVar()                                   
# name (login page) as a string when user enters it in entry box
ps_log = StringVar()                                   
# password  (login page) as a string when user enters it in entry box

Intro_login = Label(login, text="Welcome to Movie Recommender! Please login.", fg='blue', 
                    font=("arial", 30, "bold")).pack()
# the head-label of the login page
# pack essentially means printing in the centre of the page


def abt():                                              
    # initialising a method which shows a message in the login page
    tkinter.messagebox.showinfo("About", "This is a movie recommender. You must first log in to create your account. The application will guide you. Welcome!")


menu_login = Menu(login)                                    # creating menu bar in login page
login.config(menu=menu_login)

About = Menu(login)                                         # creating 'about' in menu
menu_login.add_cascade(label="About", menu=About)           # adding dropdown in about
About.add_command(label="What is this program?", command=abt)           
# when clicked, the abt command messagebox will open

userName_login = Label(login, text="Username : ", font=("arial", 16, "bold"))        
# create a first name label for login page
userName_login.place(relx=0.34, rely=0.33)

entry_userName_login = Entry(login, textvar=un_log)                                    
# entrybox for first name login page
entry_userName_login.place(relx=0.55, rely=0.33)

Name_login = Label(login, text="Name : ", font=("arial", 16, "bold"))          
# last name label login page
Name_login.place(relx=0.34, rely=0.4)

entry_Name_login = Entry(login, textvar=name_log)                                     
# entry box last name login page
entry_Name_login.place(relx=0.55, rely=0.4)

Password_login = Label(login, text="Password : ", font=("arial", 16, "bold"))            
# password label login page
Password_login.place(relx=0.34, rely=0.47)

entry_password_login = Entry(login, textvar=ps_log, show='*')                            
# entry box password login page
entry_password_login.place(relx=0.55, rely=0.47)

Quit_button_login = Button(login, text="Quit", width=12, fg='brown', command=login.quit)       
# quit button login page
Quit_button_login.place(relx=0.53, rely=0.6)

Signup_instruction = Label(login, text="If you don't have an account, please click on 'Sign Up'.", fg='black', 
                           font=("arial", 14)).place(relx=0.38, rely=0.76)
# signup text on login page

signUp_button = Button(login, text="Sign Up", width=12, fg='brown', command=signup_method)   
# signup button login page (command which will open signup window)
signUp_button.place(relx=0.46, rely=0.675)

login_button_login = Button(login, text="Login", width=12, fg='brown', command=login_database)      
# login button on login page (which will open homepage)
login_button_login.place(relx=0.39, rely=0.6)


login.mainloop()
exit(0)           


# In[ ]:




