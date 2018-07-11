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

>are you ok?  
yes . i 'm sorry . i 'm sorry .  
yes . i 'm sorry . i was dirvorced .  
yes i 'm sorry . i 'm sorry . i  
yes i 'm sorry . i did n't mean to  

>are you crazy?  
what are you going to do ? tell me .  
what are you going to do ? forget it .  
what are you going to do , dad ?!  
what are you going to do ? shoot him ?  
what are you going to do ? tell him .  

As shown above, the replies appear to have some basic logics. However, the answers and question are not quite matched. Sometimes answers are simply repeated words, which are mainly caused by unknown input words. I will test biger dataset and vocab dictionary latter and hope it will have a better performance.  

Notice: this model can't tell capital, eg. What is unknown
