# Makefile for Smart Recycling Detection System

.PHONY: setup simulator real docker-up docker-down help

help:
	@echo "Available commands:"
	@echo "  make setup       - Install dependencies"
	@echo "  make simulator   - Run in Simulator Mode (No hardware)"
	@echo "  make real        - Run in Real Mode (Connected to ESP32)"
	@echo "  make docker-up   - Start Docker services"
	@echo "  make docker-down - Stop Docker services"

setup:
	pip install -r requirements.txt

simulator:
	export HAS_HARDWARE=False && python main.py

real:
	export HAS_HARDWARE=True && python main.py

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
