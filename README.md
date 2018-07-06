# SimpleChatbot
Simple Chatbot using seq2seq model and attention   
python3.5 + tensorflow-gpu1.5.0 + nltk3.3 + tqdm4..23.4 + numpy1.14.5  

Dataset is Cornell Movie Dialogues dataset.

Reference:  
@author Conchylicultor DeepQA https://github.com/Conchylicultor/DeepQA  
@author lc222 seq2seq_chatbot_new https://github.com/lc222/seq2seq_chatbot_new

The model is traied by 100 epochs on gpu, takes about 17 hours.  

Result:  
>what time is it?  
i do n't know . it 's george .  
i do n't know . sometime . i think .  
i do n't know . sometime . maybe .  
i do n't know . sometime . over . .  
i do n't know . sometime . over . here  

>What time is it?  
just half a minute . i 'll take it .  
just half a minute . i 'll call you .  
just half a minute . i 'll be there .  
just half a minute . i 've got to .  
just half a minute . i 'll see you later .  

>Are you crazy?  
it 's all right . i do n't know .  
it 's all right . i do n't understand .  
it 's all right . i guess . yes .  
it 's all right . i guess . it 's .
it 's all right . i guess . i ...

As shown above, the replies appear to have some basic logics. However, the answers and question are not quite matched. Sometimes answers are simply repeated words, which are mainly caused by unknown input words. I will test biger dataset and vocab dictionary latter and hope it will have a better performance. 
