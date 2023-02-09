
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv(verbose=True)


class OxfordAPISettings(BaseSettings):
    accept:str = Field(..., env="ACCEPT")
    app_id:str = Field(..., env="APP_ID")
    app_key:str = Field(..., env="APP_KEY")
    url:str = Field(..., env="URL")
    strict_match:str = Field(..., env="STRICT_MATCH")

    def __str__(self):
        return self.accept + '\n' + self.app_id + '\n' + self.app_key + '\n' + self.url + '\n'

    class Config:
        env_file = "../.env"
        env_file_encoding = "utf-8"

if __name__ == '__main__':
    print(OxfordAPISettings())

