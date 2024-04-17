from django.db import models

class Student(models.Model):
    student_id = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class AttentionRecord(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    attention_span = models.FloatField()
    frame_time = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Record for {self.student.name} ({self.timestamp})"
