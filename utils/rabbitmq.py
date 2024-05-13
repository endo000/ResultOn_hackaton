import os

import pika


class RabbitMQ:
    def __init__(self, exchange="hackathon", routing_key="hackathon"):
        self.exchange = exchange
        self.routing_key = routing_key

        credentials = pika.PlainCredentials(
            os.getenv("PIKA_USER", "guest"),
            os.getenv("PIKA_PASS", "guest"),
        )
        parameters = pika.ConnectionParameters(
            host=os.getenv("PIKA_HOST", "localhost"),
            port=os.getenv("PIKA_PORT", "5672"),
            credentials=credentials,
        )
        self.connection = pika.BlockingConnection(parameters)

        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange=self.exchange, exchange_type="topic")

    def __del__(self):
        self.connection.close()

    def publish(self, message):
        self.channel.basic_publish(
            exchange=self.exchange,
            routing_key=self.routing_key,
            body=message,
        )

    def receive(self):
        result = self.channel.queue_declare("", exclusive=True)
        queue_name = result.method.queue

        self.channel.queue_bind(
            exchange=self.exchange,
            queue=queue_name,
            routing_key=self.routing_key,
        )

        print(" [*] Waiting for logs. To exit press CTRL+C")

        def callback(ch, method, properties, body):
            print(f" [x] {method.routing_key}:{body}")

        self.channel.basic_consume(
            queue=queue_name, on_message_callback=callback, auto_ack=True
        )

        self.channel.start_consuming()


if __name__ == "__main__":
    rabbit = RabbitMQ(routing_key="#")
    rabbit.receive()
