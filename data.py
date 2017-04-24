"""
**********************************************************************
*    image and input files path directories
**********************************************************************
"""
a1 = 'test1.jpg'
a2 = 'test2.jpg'
a3 = 'test3.jpg'
a4 = 'test4.jpg'
a5 = 'test5.jpg'
a6 = 'test6.jpg'

v1 = 'challenge_video.mp4'
v2 = 'project_video.mp4'
v3 = 'harder_challenge_video.mp4'

isVideo = False

if isVideo:
    video = v2
    folder_add = ''
    img_add = video
else:
    image = a1
    folder_add = 'test/'
    img_add = 'test/' + image
