# bdb_technical_test
Este repositorio contiene el código realizado para la prueba técnica del cargo científico de datos senior del Banco de Bogotá.

Si se quiere realizar la ejecución de este código ejecutar los siguientes comandos (en la carpeta que se desee ejecutar):
* crear entorno virtual: `python -m venv ./venv`
* activar entorno virtual: `.\venv\Scripts\activate`
* el comando `pip install -r requirements.txt` para instalar las librerías necesarias.
* es necesario contar con una [**api_key**](https://platform.openai.com/api-keys) de OpenAI para poder realizar las consultas.
* todo se ejecuta desde el notebook de prueba técnica.

**Advertencia:** a pesar de fijar la temperatura en 0 para el chat con el LLM es posible que los resultados de la agrupación sean ligeramente diferentes. Por ello, se dejó un json con el mapeo de las categorías.

Este experimento fue ejecutado en un computador con Intel Core I7 de 8va generación a 1.8 GHz con 4 núcleos, con una memoria RAM de 12 GB.
