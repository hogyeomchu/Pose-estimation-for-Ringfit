import pygame
import sys

def continue_or_quit():
    # Pygame 초기화
    pygame.init()

    # 화면 크기 설정
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("운동 계속 여부 확인")

    # 폰트 설정
    font = pygame.font.Font(None, 48)
    button_font = pygame.font.Font(None, 36)

    # 색상 설정
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)

    # 버튼 위치와 크기
    yes_button_rect = pygame.Rect(200, 400, 150, 60)
    no_button_rect = pygame.Rect(450, 400, 150, 60)

    # 배경 이미지 로드
    try:
        background_image = pygame.image.load("background2.jpg")
        background_image = pygame.transform.scale(background_image, (screen_width, screen_height))
    except FileNotFoundError:
        print("배경 이미지 파일을 찾을 수 없습니다. 기본 하얀 배경을 사용합니다.")
        background_image = None

    # 루프
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if yes_button_rect.collidepoint(event.pos):
                    print("Yes 선택: 운동 계속 진행")
                    pygame.quit()
                    return True
                elif no_button_rect.collidepoint(event.pos):
                    print("No 선택: 프로그램 종료")
                    pygame.quit()
                    sys.exit()

        # 배경 이미지 그리기
        if background_image:
            screen.blit(background_image, (0, 0))
        else:
            screen.fill(WHITE)

        # 질문 텍스트 렌더링
        question_text = font.render("Next Workout?", True, RED)
        screen.blit(question_text, (screen_width // 2 - question_text.get_width() // 2, 350))

        # Yes 버튼 그리기
        pygame.draw.rect(screen, GREEN, yes_button_rect)
        yes_text = button_font.render("Yes", True, BLACK)
        screen.blit(yes_text, (yes_button_rect.centerx - yes_text.get_width() // 2, 
                               yes_button_rect.centery - yes_text.get_height() // 2))

        # No 버튼 그리기
        pygame.draw.rect(screen, RED, no_button_rect)
        no_text = button_font.render("No", True, BLACK)
        screen.blit(no_text, (no_button_rect.centerx - no_text.get_width() // 2, 
                              no_button_rect.centery - no_text.get_height() // 2))

        # 화면 업데이트
        pygame.display.flip()

# 메인 실행 코드
if __name__ == "__main__":
    result = continue_or_quit()
    if result:
        print("운동을 계속 진행합니다.")
    else:
        print("운동을 종료합니다.")
