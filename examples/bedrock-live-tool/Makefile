.PHONY: run

AUDIO_IN ?= testdata/weather-ask-16k.pcm
AUDIO_OUT ?= /tmp/sonic-tool-out.pcm

run:
	go run . --audio-in $(AUDIO_IN) --audio-out $(AUDIO_OUT)
