# استخدام صورة بايثون رسمية
FROM python:3.10

# تعيين مجلد العمل
WORKDIR /code

# نسخ ملف المتطلبات وتثبيتها
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# نسخ باقي ملفات المشروع
COPY . /code

# منح صلاحية التنفيذ لسكربت التشغيل وتشغيله
RUN chmod +x /code/start.sh
CMD ["/code/start.sh"]