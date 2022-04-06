import streamlit as st
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import sys 

from sympy import false
import mrec
import radar

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_id='f6e77303443749bb809dd69d184d9703'
client_secret='100a1938944043329e9a3dbdb4936fe9'

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

face_detection = cv2.CascadeClassifier(r'haarcascade\haarcascade_frontalface_default.xml')
emotion_classifier = load_model(r'DCNN_new.h5', compile=False)
EMOTIONS = ["hapiness","sadness",'surprise','neutral']


def take_input():
    camera = cv2.VideoCapture(0)
    while True:
    # 카메라 이미지 캡처
        ret, frame = camera.read()
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1)
        if key == ord('C'):
            break
    # 컬러에서 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 프레임에서 안면인식
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30,30))
    # 이미지 공간 생성
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    # 안면인식이 가능할 경우만 감정인식 진행
    if len(faces) > 0:
        # 가장 큰 이미지 Sorting
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        # 이미지를 48x48로 사이즈 변환후 신경망에 구성
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # 감정 예측
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        
        #함수 라벨 전역변수로 변환
        global label
        label = EMOTIONS[preds.argmax()]
        
        # 라벨 지정
        cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
 
        #global emotion
        # 라벨 Output
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            #global text
            text = "{}: {:.2f}%".format(emotion, prob * 100)    
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

         # 2개 창 띄우기
        ## 이미지 디스플레이 ("감정인식")
        ## 감정확률 디스플레이 ("확률")
            cv2.imshow('Emotion Recognition', frame)
            cv2.imshow('Probabilities', canvas)


    
        showPic = cv2.imwrite("photo.jpg",frame)
        print(showPic) 
        # 프로그램 클리어 및 창 닫기
        camera.release()
        cv2.destroyAllWindows()

header  = st.container()
inp = st.container()
pred = st.container()

with header:
    st.image('bgc1.jpeg')
    st.title('감정 분석 음악 추천 서비스')
    st.markdown('**캡처된 당신의 얼굴 사진을 분석 후 감정에 따른 음악 추천을 드립니다.**')

with inp:
    st.title("당신의 얼굴 사진을 캡처합니다.")
    st.markdown("**캡처 진행을 하시려면 SHIFT+C를 눌러주세요!**")
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    take_input()

with pred:
    st.title("추천드릴 음악을 찾아보겠습니다! 잠시만 기다려주세요!")
    
    predictions = str(label)

    if(predictions == 'hapiness'):
 
        st.markdown("**기분이 좋아 보이시군요! 당신을 행복하게해 줄 노래입니다! 기분 좋은 하루 보내세요^^ - 당신의 AI DJ**")
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
    
    elif (predictions == 'sadness'):
      
        st.markdown("**흠... 기분이 안 좋으신 하루 같군요... 당신을 위로해 줄 노래입니다! 기분 좋은 하루 보내세요^^ - 당신의 AI DJ**")
        playlist_id = '7LadVOHuQzetJSCxL1zlZE'
    
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
        

    elif (predictions == 'surprise'):
      
        st.markdown("**많이 놀라셨죠?... 당신을 진정시켜줄 노래입니다! 기분 좋은 하루 보내세요^^ - 당신의 AI DJ**")
        playlist_id = '13ygyCntSq0BK0b3CgZQ3a'

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
        predictions == 'neutral'

        st.markdown("**흠... 별일없으세여?... 당신의 하루를 즐겁게 열어줄 노래입니다. 기분 좋은 하루 보내세요^^ - 당신의 AI DJ**")

search_choices = ['노래/음원', '아티스트', '앨범']
search_selected = st.sidebar.selectbox("원하는 검색 선택: ", search_choices)

search_keyword = st.text_input(search_selected + " (키워드 검색)")
button_clicked = st.button("검색")


search_results = []
tracks = []
artists = []
albums = []
if search_keyword is not None and len(str(search_keyword)) > 0:
    if search_selected == '노래/음원':
        st.write("노래/음원 검색 결과")
        tracks = sp.search(q='track:'+ search_keyword,type='track', limit=20)
        tracks_list = tracks['tracks']['items']
        if len(tracks_list) > 0:
            for track in tracks_list:
                #st.write(track['name'] + " - By - " + track['artists'][0]['name'])
                search_results.append(track['name'] + " - By - " + track['artists'][0]['name'])
        
    elif search_selected == '아티스트':
        st.write("아티스트 검색 결과")
        artists = sp.search(q='artist:'+ search_keyword,type='artist', limit=20)
        artists_list = artists['artists']['items']
        if len(artists_list) > 0:
            for artist in artists_list:
                # st.write(artist['name'])
                search_results.append(artist['name'])
        
    if search_selected == '앨범':
        st.write("앨범 검색 결과")
        albums = sp.search(q='album:'+ search_keyword,type='album', limit=20)
        albums_list = albums['albums']['items']
        if len(albums_list) > 0:
            for album in albums_list:
                # st.write(album['name'] + " - By - " + album['artists'][0]['name'])
                # print("Album ID: " + album['id'] + " / Artist ID - " + album['artists'][0]['id'])
                search_results.append(album['name'] + " - By - " + album['artists'][0]['name'])

selected_album = None
selected_artist = None
selected_track = None
if search_selected == '노래/음원':
    selected_track = st.selectbox("원하는 노래/음원 선택: ", search_results)
elif search_selected == '아티스트':
    selected_artist = st.selectbox("원하는 아티스트 선택: ", search_results)
elif search_selected == '앨범':
    selected_album = st.selectbox("원하는 앨범 선택: ", search_results)


if selected_track is not None and len(tracks) > 0:
    tracks_list = tracks['tracks']['items']
    track_id = None
    if len(tracks_list) > 0:
        for track in tracks_list:
            str_temp = track['name'] + " - By - " + track['artists'][0]['name']
            if str_temp == selected_track:
                track_id = track['id']
                track_album = track['album']['name']
                img_album = track['album']['images'][1]['url']
                mrec.save_album_image(img_album, track_id)
    selected_track_choice = None            
    if track_id is not None:
        image = mrec.get_album_mage(track_id)
        st.image(image)
        track_choices = ['음원 분석 속성', '유사 음원 추천']
        selected_track_choice = st.sidebar.selectbox('음원 분석 선택: ', track_choices)        
        if selected_track_choice == '음원 분석 속성':
            track_features  = sp.audio_features(track_id) 
            df = pd.DataFrame(track_features, index=[0])
            df_features = df.loc[: ,['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']]
            st.dataframe(df_features)
            radar.feature_plot(df_features)
        elif selected_track_choice == '유사 음원 추천':
            token = mrec.get_token(client_id, client_secret)
            similar_songs_json = mrec.get_track_recommendations(track_id, token)
            recommendation_list = similar_songs_json['tracks']
            recommendation_list_df = pd.DataFrame(recommendation_list)
            # st.dataframe(recommendation_list_df)
            recommendation_df = recommendation_list_df[['name', 'explicit', 'duration_ms', 'popularity']]
            st.dataframe(recommendation_df)
            # st.write("Recommendations....")
            mrec.song_recommendation_vis(recommendation_df)
            
    else:
        st.write("리스트에서 음원을 선택해주세요")       

elif selected_album is not None and len(albums) > 0:
    albums_list = albums['albums']['items']
    album_id = None
    album_uri = None    
    album_name = None
    if len(albums_list) > 0:
        for album in albums_list:
            str_temp = album['name'] + " - By - " + album['artists'][0]['name']
            if selected_album == str_temp:
                album_id = album['id']
                album_uri = album['uri']
                album_name = album['name']
    if album_id is not None and album_uri is not None:
        st.write("앨범의 모든 음원 수집:" + album_name)
        album_tracks = sp.album_tracks(album_id)
        df_album_tracks = pd.DataFrame(album_tracks['items'])
        # st.dataframe(df_album_tracks)
        df_tracks_min = df_album_tracks.loc[:,
                        ['id', 'name', 'duration_ms', 'explicit', 'preview_url']]
        # st.dataframe(df_tracks_min)
        for idx in df_tracks_min.index:
            with st.container():
                col1, col2, col3, col4 = st.columns((4,4,1,1))
                col11, col12 = st.columns((8,2))
                col1.write(df_tracks_min['id'][idx])
                col2.write(df_tracks_min['name'][idx])
                col3.write(df_tracks_min['duration_ms'][idx])
                col4.write(df_tracks_min['explicit'][idx])   
                if df_tracks_min['preview_url'][idx] is not None:
                    col11.write(df_tracks_min['preview_url'][idx])  
                    with col12:   
                        st.audio(df_tracks_min['preview_url'][idx], format="audio/mp3")                            
                        
                        
if selected_artist is not None and len(artists) > 0:
    artists_list = artists['artists']['items']
    artist_id = None
    artist_uri = None
    selected_artist_choice = None
    if len(artists_list) > 0:
        for artist in artists_list:
            if selected_artist == artist['name']:
                artist_id = artist['id']
                artist_uri = artist['uri']
    
    if artist_id is not None:
        artist_choice = ['앨범', '상위 음원']
        selected_artist_choice = st.sidebar.selectbox('아티스트 선택', artist_choice)
                
    if selected_artist_choice is not None:
        if selected_artist_choice == 'Albums':
            artist_uri = 'spotify:artist:' + artist_id
            album_result = sp.artist_albums(artist_uri, album_type='album') 
            all_albums = album_result['items']
            col1, col2, col3 = st.columns((6,4,2))
            for album in all_albums:
                col1.write(album['name'])
                col2.write(album['release_date'])
                col3.write(album['total_tracks'])
        elif selected_artist_choice == 'Top Songs':
            artist_uri = 'spotify:artist:' + artist_id
            top_songs_result = sp.artist_top_tracks(artist_uri)
            for track in top_songs_result['tracks']:
                with st.container():
                    col1, col2, col3, col4 = st.columns((4,4,2,2))
                    col11, col12 = st.columns((10,2))
                    col21, col22 = st.columns((11,1))
                    col31, col32 = st.columns((11,1))
                    col1.write(track['id'])
                    col2.write(track['name'])
                    if track['preview_url'] is not None:
                        col11.write(track['preview_url'])  
                        with col12:   
                            st.audio(track['preview_url'], format="audio/mp3")  
                    with col3:
                        def feature_requested():
                            track_features  = sp.audio_features(track['id']) 
                            df = pd.DataFrame(track_features, index=[0])
                            df_features = df.loc[: ,['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']]
                            with col21:
                                st.dataframe(df_features)
                            with col31:
                                radar.feature_plot(df_features)
                            
                        feature_button_state = st.button('음원 분석 속성', key=track['id'], on_click=feature_requested)
                    with col4:
                        def similar_songs_requested():
                            token = mrec.get_token(client_id, client_secret)
                            similar_songs_json = mrec.get_track_recommendations(track['id'], token)
                            recommendation_list = similar_songs_json['tracks']
                            recommendation_list_df = pd.DataFrame(recommendation_list)
                            recommendation_df = recommendation_list_df[['name', 'explicit', 'duration_ms', 'popularity']]
                            with col21:
                                st.dataframe(recommendation_df)
                            with col31:
                                mrec.song_recommendation_vis(recommendation_df)

                        similar_songs_state = st.button('유사 음원 추천', key=track['id'], on_click=similar_songs_requested)
                    st.write('----')