
from pydantic import BaseSettings, Field

class OxfordAPISettings(BaseSettings):
    accept:str = Field(..., env="ACCEPT")
    app_id:str = Field(..., env="APP_ID")
    app_key:str = Field(..., env="APP_KEY")
    url:str = Field(..., env="URL")
    strict_match:str = Field(..., env="STRICT_MATCH")

    class Config:
        env_file = "../.env"

if __name__ == '__main__':
    print(OxfordAPISettings().strict_match)

