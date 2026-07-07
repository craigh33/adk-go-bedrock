// S3 artifact service example for adk-go-bedrock: wires the artifact/s3
// [artifact.Service] into an ADK runner so agent/tool artifacts persist in
// Amazon S3 instead of process memory, then saves and reloads an artifact
// directly to show the round trip.
//
// Authenticate with AWS using the default credential chain and set:
//
//	ARTIFACT_S3_BUCKET  (required) bucket to store artifacts in
//	ARTIFACT_S3_PREFIX  (optional) key prefix, e.g. "adk-artifacts"
//	AWS_REGION          (optional) region for the S3 client
//
// Run:
//
//	go run ./examples/bedrock-artifact-s3
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/genai"

	s3artifact "github.com/craigh33/adk-go-bedrock/artifact/s3"
)

func main() {
	ctx := context.Background()

	bucket := os.Getenv("ARTIFACT_S3_BUCKET")
	if bucket == "" {
		log.Fatal("ARTIFACT_S3_BUCKET is required")
	}

	awsCfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Fatalf("load AWS config: %v", err)
	}

	svc, err := s3artifact.NewService(s3.NewFromConfig(awsCfg), s3artifact.Config{
		Bucket:    bucket,
		KeyPrefix: os.Getenv("ARTIFACT_S3_PREFIX"),
	})
	if err != nil {
		log.Fatalf("create artifact service: %v", err)
	}

	// The service plugs straight into an ADK runner:
	//
	//	run, err := runner.New(runner.Config{
	//		AppName:         "my-app",
	//		Agent:           myAgent,
	//		SessionService:  session.InMemoryService(),
	//		ArtifactService: svc,
	//	})
	//
	// Below, the same operations agents/tools perform via ctx.Artifacts(),
	// called directly for demonstration.

	saved, err := svc.Save(ctx, &artifact.SaveRequest{
		AppName: "artifact-s3-example", UserID: "demo-user", SessionID: "demo-session",
		FileName: "greeting.txt",
		Part:     genai.NewPartFromText("hello from S3-backed artifacts"),
	})
	if err != nil {
		log.Fatalf("save artifact: %v", err)
	}
	fmt.Printf("saved greeting.txt version %d\n", saved.Version)

	loaded, err := svc.Load(ctx, &artifact.LoadRequest{
		AppName: "artifact-s3-example", UserID: "demo-user", SessionID: "demo-session",
		FileName: "greeting.txt", // Version 0 = latest
	})
	if err != nil {
		log.Fatalf("load artifact: %v", err)
	}
	fmt.Printf("loaded: %s\n", loaded.Part.InlineData.Data)
}
