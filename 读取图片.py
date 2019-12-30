import requests
from PIL import Image
captcha_url = "https://file.xintujing.cn/activity/2042764520.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=qidaoyun%2F20191117%2F%2Fs3%2Faws4_request&X-Amz-Date=20191117T053153Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=327f15f4a44785874fc6109fa75e66c178f0a5a79634395397234cc516eb5409"
r = requests.get(captcha_url)
with open('captcha.jpg', 'wb') as f:
    f.write(r.content)
    f.close()

    im = Image.open('captcha.jpg')
    im.show()



