        else:
            print(f"⚠️ [CLOUDINARY] Deletion response: {result}")
            return {"message": "Cloudinary reported: " + str(result.get("result"))}, 200

    except Exception as e:
        print(f"❌ [CLOUDINARY-ERROR] Failed to delete video: {e}")
        return {"error": str(e)}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
