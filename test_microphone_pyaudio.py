"""
Test script to list and test microphones using PyAudio.
Use this to find the device index of the microphone you need for speech recognition.

Usage:
  python test_microphone_pyaudio.py              # List all input devices
  python test_microphone_pyaudio.py --test 1    # Test record + speech recognition on device index 1
"""
import argparse
import sys

try:
    import pyaudio
except ImportError:
    print("PyAudio no está instalado. Instálalo con: pip install pyaudio")
    sys.exit(1)


def list_input_devices():
    """List all audio input devices with index, name, sample rate and channels."""
    p = pyaudio.PyAudio()
    print("\n" + "=" * 70)
    print("DISPOSITIVOS DE ENTRADA (MICRÓFONOS)")
    print("=" * 70)
    print(f"{'Índice':<8} {'Nombre':<45} {'Sample rate':<12} {'Canales':<8}")
    print("-" * 70)

    input_devices = []
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                name = (info.get("name") or "Sin nombre")[:44]
                default_sr = int(info.get("defaultSampleRate", 0))
                channels = info.get("maxInputChannels", 0)
                print(f"{i:<8} {name:<45} {default_sr:<12} {channels:<8}")
                input_devices.append((i, info))
        except Exception as e:
            print(f"{i:<8} (error: {e})")

    p.terminate()
    print("-" * 70)
    print(f"Total: {len(input_devices)} dispositivo(s) de entrada.\n")
    print("Para probar un micrófono: python test_microphone_pyaudio.py --test <índice>")
    return input_devices


def test_device(device_index: int):
    """Record from the given device using SpeechRecognition (same stack as the app) and run Google recognition."""
    try:
        import speech_recognition as sr
    except ImportError:
        print("speech_recognition no instalado. Instálalo: pip install SpeechRecognition")
        return

    p = pyaudio.PyAudio()
    try:
        info = p.get_device_info_by_index(device_index)
        if info.get("maxInputChannels", 0) == 0:
            print(f"El dispositivo {device_index} no es de entrada.")
            return
        name = info.get("name", "Sin nombre")
    finally:
        p.terminate()

    print(f"\nProbando dispositivo {device_index}: {name}")
    print("Escuchando hasta 5 segundos... habla ahora.\n")

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.2
    recognizer.dynamic_energy_threshold = True

    try:
        microphone = sr.Microphone(device_index=device_index)
    except Exception as e:
        print(f"Error al abrir el micrófono {device_index}: {e}")
        return

    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
    except Exception as e:
        print(f"Error al ajustar ruido: {e}")

    try:
        with microphone as source:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        text = recognizer.recognize_google(audio, language="es-ES")
        print(f"Reconocido (Google, es-ES): \"{text}\"")
    except sr.WaitTimeoutError:
        print("No se detectó voz a tiempo.")
    except sr.UnknownValueError:
        print("No se pudo entender el audio.")
    except sr.RequestError as e:
        print(f"Error del servicio de reconocimiento: {e}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Listar o probar micrófonos con PyAudio")
    parser.add_argument(
        "--test",
        type=int,
        metavar="INDEX",
        help="Índice del dispositivo a probar (graba 3 s y opcionalmente reconoce con Google)",
    )
    args = parser.parse_args()

    if args.test is not None:
        test_device(args.test)
    else:
        list_input_devices()


if __name__ == "__main__":
    main()
