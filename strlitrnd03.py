import streamlit as st
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import capture2
import random

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_credentials_manager = SpotifyClientCredentials(client_id='f6e77303443749bb809dd69d184d9703', client_secret='100a1938944043329e9a3dbdb4936fe9')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

header  = st.container()
inp = st.container()
pred = st.container()

with header:
    st.image('bgc1.png')
    st.title('감정 분석 음악 추천 서비스')
    st.markdown('**캡처된 당신의 얼굴 사진을 분석 후 감정에 따른 음악 추천을 드립니다.**')

with inp:
    st.title("당신의 얼굴 사진을 캡처합니다.")
    st.markdown("**캡처 진행을 하시려면 SHIFT+C를 눌러주세요!**")
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    capture2.take_input()

with pred:
    st.title("추천드릴 음악을 찾아보겠습니다! 잠시만 기다려주세요!")
    ##img = cv2.imread('photo.jpg')

    ##plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    

    
    predictions = str(capture2.label)

    if(predictions == 'sadness'):

        st.markdown("**흠... 기분이 안 좋으신 하루 같군요... 당신을 위로해 줄 노래입니다! 기분 좋은 하루 보내세요^^ - 당신의 AI DJ**")
        playlist_id = '37i9dQZF1DX5a7mln8z0Su'
    
        def get_track_ids1(playlist_id):
            music_id_list = []
            playlist = sp.playlist(playlist_id)

            for item in playlist['tracks']['items']:
                music_track = item['track']
                music_id_list.append(music_track['id'])
            return music_id_list

        track_ids = get_track_ids1(playlist_id)

        for i in range(5):

            random.shuffle(track_ids)

            my_html = '<iframe src="https://open.spotify.com/embed/track/{}" width="300" height="100" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'.format(track_ids[0])

            st.markdown(my_html, unsafe_allow_html=True)
    
    else:    
        predictions != 'sadness'
        
        st.markdown("**오호! 기분이 좋으신 하루 같네요? 행복한 노래를 추천드리겠습니다!^^ - 당신의 AI DJ**")
        playlist_id = '37i9dQZF1DX2l1BM6ggRui'
    
        def get_track_ids2(playlist_id):
            music_id_list = []
            playlist = sp.playlist(playlist_id)

            for item in playlist['tracks']['items']:
                music_track = item['track']
                music_id_list.append(music_track['id'])
            return music_id_list

        track_ids = get_track_ids2(playlist_id)
        
        for i in range(5):

            random.shuffle(track_ids)

            my_html = '<iframe src="https://open.spotify.com/embed/track/{}" width="300" height="100" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'.format(track_ids[0])

            st.markdown(my_html, unsafe_allow_html=True)