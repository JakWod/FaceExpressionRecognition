import cv2
import os
import numpy as np
import random
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from fer import FER

def get_unique_folder_name(base_folder):
    folder_name = base_folder
    counter = 1
    while os.path.exists(folder_name):
        folder_name = f"{base_folder}_{counter}"
        counter += 1
    return folder_name

def apply_random_transformations(face_img):
    if random.choice([True, False]):
        max_shift = 20
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        face_img = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
    if random.choice([True, False]):
        brightness_factor = random.uniform(0.5, 1.5)
        face_img = cv2.convertScaleAbs(face_img, alpha=brightness_factor, beta=0)
    if random.choice([True, False]):
        zoom_factor = random.uniform(1.1, 1.5)
        width, height = int(face_img.shape[1] * zoom_factor), int(face_img.shape[0] * zoom_factor)
        face_img = cv2.resize(face_img, (width, height))
        face_img = face_img[:face_img.shape[0], :face_img.shape[1]]
    return face_img

def input_emotions():
    print("Wprowadź emocje ręcznie (procenty dla każdej kategorii, suma musi wynosić 100%).")
    emotion_categories = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    user_emotions = {}
    total = 0

    for category in emotion_categories:
        while True:
            try:
                value = float(input(f"{category}: "))
                if value < 0 or value > 100:
                    print("Wartość musi być w przedziale od 0 do 100.")
                    continue
                total += value
                if total > 100:
                    print("Łączna suma przekroczyła 100%. Wprowadź wartości ponownie.")
                    total -= value
                    continue
                user_emotions[category.lower()] = value / 100
                break
            except ValueError:
                print("Wprowadź poprawną liczbę.")
    
    if total != 100:
        print(f"Suma wynosi {total}%. Spróbuj ponownie.")
        return input_emotions()
    
    return user_emotions

def edit_detected_emotions(emotions):
    print("Wykryto następujące emocje:")
    for emotion, score in emotions.items():
        print(f"{emotion.capitalize()}: {score * 100:.2f}%")
    
    print("Czy chcesz edytować te emocje? (t/n)")
    edit_choice = input().lower()
    
    if edit_choice == "t":
        return input_emotions()
    else:
        return emotions

def draw_text_with_outline(image, text, position, font, font_scale, font_thickness, text_color, outline_color):
    x, y = position
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        cv2.putText(image, text, (x + dx, y + dy), font, font_scale, outline_color, font_thickness + 1, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

def process_image(img, face_classifier, emotion_detector, unique_faces_folder):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    
    accepted_faces = []
    for i, (x, y, w, h) in enumerate(faces):
        face_img = img[y:y + h, x:x + w]
        cv2.imshow(f"Twarz {i + 1}", face_img)
        print(f"Czy ten kwadrat to twarz? Naciśnij 'y' (tak) lub 'n' (nie).")

        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            accepted_faces.append((x, y, w, h))
            print(f"Twarz {i + 1} zaakceptowana.")
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            print(f"Twarz {i + 1} odrzucona.")
        cv2.destroyWindow(f"Twarz {i + 1}")

    process_faces(img, accepted_faces, emotion_detector, unique_faces_folder)
    return img

def process_faces(img, faces, emotion_detector, unique_faces_folder):
    for i, (x, y, w, h) in enumerate(faces):
        face_img = img[y:y + h, x:x + w]
        
        face_folder = os.path.join(unique_faces_folder, f"face_{i + 1}")
        os.makedirs(face_folder, exist_ok=True)
        
        face_output_path = os.path.join(face_folder, f"face_{i + 1}.jpg")
        cv2.imwrite(face_output_path, face_img)
        
        transformed_emotions = []
        for j in range(50):
            transformed_face = apply_random_transformations(face_img)
            face_img_rgb = cv2.cvtColor(transformed_face, cv2.COLOR_BGR2RGB)
            detected_emotions = emotion_detector.detect_emotions(face_img_rgb)
            
            if detected_emotions:
                emotion_scores = detected_emotions[0]["emotions"]
                transformed_emotions.append(emotion_scores)
            else:
                transformed_emotions.append({"neutral": 1.0})

        avg_emotions = {emotion: 0 for emotion in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]}
        
        for emotions in transformed_emotions:
            for emotion, score in emotions.items():
                avg_emotions[emotion] += score

        num_transforms = len(transformed_emotions)
        for emotion in avg_emotions:
            avg_emotions[emotion] /= num_transforms

        print(f"Średnie emocje dla twarzy {i + 1} (przekształcone):")
        for emotion, score in avg_emotions.items():
            print(f"{emotion.capitalize()}: {score * 100:.2f}%")

        avg_emotions = edit_detected_emotions(avg_emotions)

        label_texts = [f"ID: {i + 1}"]
        for emotion, avg_score in avg_emotions.items():
            label_texts.append(f"{emotion.capitalize()}: {avg_score * 100:.2f}%")
        
        for line_num, label_text in enumerate(label_texts):
            label_y = y + h + 20 + (line_num * 20)
            draw_text_with_outline(img, label_text, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1, (0, 255, 0), (0, 0, 0))

def process_webcam(face_classifier, emotion_detector):
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Nie można otworzyć kamery")
        return

    # Zwiększamy rozdzielczość kamery dla lepszej detekcji
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Naciśnij 'q' aby zakończyć nagrywanie")
    
    # Słownik do śledzenia ID twarzy
    face_trackers = {}
    next_face_id = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Przeskalowanie obrazu dla szybszej detekcji
        scaling_factor = 1.0  # Możesz zmniejszyć do 0.5 jeśli potrzebujesz większej wydajności
        frame_scaled = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)
        
        gray_image = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2GRAY)
        
        # Dostrajamy parametry detekcji twarzy
        faces = face_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Przeskalowanie współrzędnych twarzy z powrotem do oryginalnego rozmiaru
        faces = [(int(x/scaling_factor), int(y/scaling_factor), 
                 int(w/scaling_factor), int(h/scaling_factor)) for (x, y, w, h) in faces]

        # Słownik aktualnie wykrytych twarzy
        current_faces = {}
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            
            # Sprawdzenie czy twarz ma odpowiedni rozmiar
            if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                continue

            try:
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                detected_emotions = emotion_detector.detect_emotions(face_img_rgb)
                
                if detected_emotions:
                    # Rysowanie prostokąta wokół twarzy z grubszą linią
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                   
                    emotions = detected_emotions[0]["emotions"]
                    
                    # Przygotowanie tekstu dla wszystkich emocji
                    label_texts = []
                    for emotion, score in emotions.items():
                        label_texts.append(f"{emotion.capitalize()}: {score * 100:.1f}%")

                    # Obliczanie wysokości tła
                    
                    
                    # Rysowanie tekstu
                    for line_num, label_text in enumerate(label_texts):
                        label_y = y + h + 20 + (line_num * 20)
                        draw_text_with_outline(
                            frame, label_text,
                            (x + 5, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, 1,
                            (0, 255, 0),
                            (0, 0, 0)
                        )
                    
                    current_faces[(x, y, w, h)] = emotions

            except Exception as e:
                print(f"Błąd podczas przetwarzania twarzy: {str(e)}")
                continue

        

        # Wyświetlanie z pełnym rozmiarem okna
        cv2.namedWindow("Detekcja Emocji na Żywo", cv2.WINDOW_NORMAL)
        cv2.imshow("Detekcja Emocji na Żywo", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Opcjonalnie: możliwość zapisania klatki
            cv2.imwrite("emotion_detection_snapshot.jpg", frame)
            print("Zapisano zrzut ekranu")

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Inicjalizacja klasyfikatora twarzy z większą dokładnością
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    # Inicjalizacja detektora emocji
    emotion_detector = FER(mtcnn=True)  # Używamy MTCNN dla lepszej detekcji

    print("Wybierz źródło obrazu:")
    print("1. Zdjęcie")
    print("2. Kamera internetowa")
    
    while True:
        choice = input("Twój wybór (1 lub 2): ")
        if choice in ['1', '2']:
            break
        print("Nieprawidłowy wybór. Wybierz 1 lub 2.")

    if choice == '1':
        Tk().withdraw()
        print("Wybierz obraz wejściowy.")
        image_path = askopenfilename(filetypes=[("Pliki graficzne", "*.jpg *.jpeg *.png")])
        
        if not image_path:
            print("Nie wybrano pliku. Program zakończy działanie.")
            return

        img = cv2.imread(image_path)
        if img is None:
            print("Nie udało się wczytać obrazu. Sprawdź, czy plik wejściowy jest prawidłowy.")
            return

        faces_folder = os.path.join(script_dir, 'faces')
        if not os.path.exists(faces_folder):
            os.makedirs(faces_folder)

        unique_faces_folder = get_unique_folder_name(os.path.join(faces_folder, 'face'))
        os.makedirs(unique_faces_folder)
        print(f"Tworzenie folderu twarzy: {unique_faces_folder}")

        processed_img = process_image(img, face_classifier, emotion_detector, unique_faces_folder)
        
        output_image_path = os.path.join(unique_faces_folder, "final_image.jpg")
        cv2.imwrite(output_image_path, processed_img)
        print(f"Obraz z etykietami został zapisany jako: {output_image_path}")

        cv2.imshow("Final Image with Labels", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass
    else:
        process_webcam(face_classifier, emotion_detector)

if __name__ == "__main__":
    main()