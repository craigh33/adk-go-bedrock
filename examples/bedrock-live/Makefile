.PHONY: run

AUDIO_IN ?= testdata/hello-16k.pcm
AUDIO_OUT ?= /tmp/sonic-out.pcm

run:
	go run . --audio-in $(AUDIO_IN) --audio-out $(AUDIO_OUT)
