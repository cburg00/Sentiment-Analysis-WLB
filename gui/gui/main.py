from customtkinter import *
from DecisionTree import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        label_btn.configure(text=f'{file_path}')
        global Decision_Tree 
        Decision_Tree = DecisionTree(file_path)

def print_data():
    print(Decision_Tree)

def generate_tree():
    model = Decision_Tree.train()
    fig = Decision_Tree.plot(model)
    canvas = FigureCanvasTkAgg(fig, master=app)
    canvas.draw()
    canvas.get_tk_widget().place(relx=0.4, rely=0.5, anchor='w')

app = CTk()
app.geometry("1000x800")

app.title('Work Life Balance Analysis')

set_appearance_mode('dark')

btn_btn = CTkButton(master=app, text="Choose File", corner_radius=32, command=select_file)
label_btn = CTkLabel(master=app, text='select file', corner_radius=32)
print_btn = CTkButton(master=app, text="Print Dataset", corner_radius=32, command=print_data)
plot_btn = CTkButton(master=app, text="Plot Tree", corner_radius=32, command=generate_tree)

btn_btn.place(relx=0.25, rely=0.5, anchor='center')
label_btn.place(relx=0.25, rely=0.3, anchor='center')
print_btn.place(relx=0.25, rely=0.6, anchor='center')
plot_btn.place(relx=0.25, rely=0.4, anchor='center')

app.mainloop()



